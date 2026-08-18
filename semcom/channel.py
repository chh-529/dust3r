"""
Channel models for SemCom. AWGN only (see NOTE at the bottom on extending to fading).

Single user (point to point):

    y = x + n                             shape (*batch, L) -> (*batch, L)

Multi user uplink (superposition / non-orthogonal multiple access):

    y = sum_k (a_k * h_k * x_k) + n       shape (*d1, n_tx, *d2, L) -> (*d1, *d2, L)

    a_k : power_alloc[k]**0.5  -- power-domain NOMA, x_k is always unit power
    h_k : channel_gain[k]      -- heterogeneous-SNR
"""

from typing import List, Optional, Sequence, Tuple
import torch
from .utils import get_class_str, signal_power


def make_complex_gaussian_noise(signal: torch.Tensor,
                                noise_power: torch.Tensor) -> torch.Tensor:
    """
    Draw n ~ CN(0, noise_power), circularly symmetric complex Gaussian.

    Args:
        signal: complex tensor;
        noise_power: sigma^2, a real tensor broadcastable to `signal`'s shape.

    Returns:
        complex noise tensor shaped like `signal`, with E[|n|^2] == noise_power.
    """
    std = torch.sqrt(noise_power / 2.0)
    noise_real = torch.randn_like(signal.real) * std
    noise_imag = torch.randn_like(signal.imag) * std
    return torch.complex(noise_real, noise_imag)


def make_awgn_noise(signal: torch.Tensor,
                    snr_db: float | torch.Tensor,
                    reference_power: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Draw circularly symmetric complex Gaussian noise for `signal`

    Args:
        signal: complex tensor of any shape;
        snr_db: SNR in dB. A float, or a real tensor broadcastable to `signal`'s shape
        reference_power: the signal power the SNR is measured against, a real tensor
            broadcastable to `signal`'s shape. If None, it is measured from `signal`
            itself, over the last dimension.

    Returns:
        complex noise tensor with the same shape / dtype / device as `signal`.
    """
    if not signal.is_complex():
        raise TypeError('make_awgn_noise() expects a complex signal')

    if reference_power is None:
        reference_power = signal_power(signal, keepdim=True)
    reference_power = reference_power.detach().to(signal.device)

    if not isinstance(snr_db, torch.Tensor):
        snr_db = torch.tensor(snr_db, dtype=reference_power.dtype, device=signal.device)
    snr_db = snr_db.to(device=signal.device, dtype=reference_power.dtype)

    snr_linear = torch.pow(10.0, snr_db / 10.0)
    return make_complex_gaussian_noise(signal, reference_power / snr_linear)


class Channel:
    """
    Base class. A channel is a stateless-ish transform on a signal tensor; it is a plain
    object rather than an nn.Module because it holds no learnable parameters, and its
    only tensor state (snr_db) is small and moved to the signal's device on demand.
    """

    def interfere(self, signal: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # must dispatch through self, so subclasses' interfere() is the one that runs
        return self.interfere(*args, **kwargs)


class SingleChannel(Channel):
    """One transmitter, one receiver."""

    def interfere(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: complex tensor of shape (*batch, L)
        Returns:
            complex tensor of shape (*batch, L)
        """
        raise NotImplementedError()


class MultiUplinkChannel(Channel):
    """Many transmitters, one receiver; the receiver sees the superposition."""

    def interfere(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
        Args:
            signal: complex tensor of shape (*d1, n_tx, *d2, L), the user dimension
                sitting at `user_dim_index`
        Returns:
            complex tensor of shape (*d1, *d2, L) -- the user dimension is consumed
        """
        raise NotImplementedError()


class AWGNSingleChannel(SingleChannel):
    """
    y = x + n,  n ~ CN(0, sigma^2),  sigma^2 = E[|x|^2] / 10^(snr_db/10)

    The SNR is measured per batch element against the actual transmitted power, so this
    stays correct whether or not the caller power-normalized beforehand. It is still
    strongly recommended to power_normalize() the encoder output, otherwise the *relative*
    power between users in the multi-user case is undefined.
    """

    def __init__(self, snr_db: float | torch.Tensor, keep_last_noise: bool = False):
        """
        Args:
            snr_db: signal to noise ratio in dB.
            keep_last_noise: store the last drawn noise on CPU, retrievable via
                get_last_noise(). Off by default -- the signals here are large
                (batch x tens of thousands of symbols) and copying them every forward
                pass costs real time and memory. Turn on only to reproduce an experiment.
        """
        self.snr_db = snr_db
        self.keep_last_noise = keep_last_noise
        self.last_noise: Optional[torch.Tensor] = None

    def __str__(self):
        return get_class_str(self, snr_db=self.snr_db)

    __repr__ = __str__

    def interfere(self, signal: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            signal: complex tensor of shape (*batch, L)
            noise: optionally supply the noise instead of drawing it, for reproducing a
                previous run. Must broadcast to `signal`'s shape.
        Returns:
            complex tensor of shape (*batch, L)
        """
        if noise is None:
            noise = make_awgn_noise(signal, self.snr_db)
        else:
            noise = noise.detach().to(signal.device)

        if self.keep_last_noise:
            self.last_noise = noise.detach().to('cpu')
        return signal + noise

    def get_last_noise(self) -> Optional[torch.Tensor]:
        """The noise used by the last interfere() call (CPU), or None if not kept."""
        return self.last_noise


class AWGNMultiUplinkChannel(MultiUplinkChannel):
    """
    y = sum_k (a_k * h_k * x_k) + n,   n ~ CN(0, sigma^2)

    a_k : power_alloc[k]**0.5  -- power-domain NOMA 
    h_k : channel_gain[k]      -- heterogeneous-SNR

    sigma^2 comes from exactly one of two places (mutually exclusive):
        snr_db       measured against the superimposed signal each forward
        noise_power  a fixed constant, not measured
    """

    def __init__(self,
                 n_tx: int,
                 snr_db: Optional[float] = None,
                 power_alloc: Optional[Sequence[float]] = None,
                 channel_gain: Optional[Sequence[float]] = None,
                 noise_power: Optional[float] = None,
                 keep_last_noise: bool = False):
        """
        Args:
            n_tx:               number of transmitters (users) that superimpose.
            snr_db:             SNR in dB against the measured superimposed signal.
            power_alloc:        a_k^2, the NOMA power split.
            channel_gain:       h_k, per-user channel gain (real amplitude; phase 0,
                                since there is no equalizer downstream to undo it).
            noise_power:        sigma^2, used as-is every forward instead of snr_db.
            keep_last_noise:    see AWGNSingleChannel.
        """
        if n_tx < 1:
            raise ValueError(f'{n_tx = } must be >= 1')
        if (snr_db is None) == (noise_power is None):
            raise ValueError('give exactly one of snr_db, noise_power')
        if isinstance(snr_db, torch.Tensor) and snr_db.numel() > 1:
            raise ValueError('snr_db must be a scalar; for per-user target SNR use '
                             'AWGNMultiUplinkChannel.make_channel_gain()')

        self.n_tx = n_tx
        self.snr_db = None if snr_db is None else float(snr_db)
        self.noise_power = None if noise_power is None else float(noise_power)
        self.power_alloc = self._as_vector(power_alloc, n_tx, default=1.0)
        self.channel_gain = self._as_vector(channel_gain, n_tx, default=1.0)
        self.keep_last_noise = keep_last_noise
        self.last_noise: Optional[torch.Tensor] = None
        self.last_effective_gain: Optional[torch.Tensor] = None

    @staticmethod
    def _as_vector(x, n: int, default: float) -> torch.Tensor:
        if x is None:
            return torch.full((n,), default, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.numel() != n:
            raise ValueError(f'expected length {n}, got {x.numel()}')
        return x

    def __str__(self):
        return get_class_str(self, n_tx=self.n_tx, snr_db=self.snr_db,
                             noise_power=self.noise_power,
                             power_alloc=self.power_alloc.tolist(),
                             channel_gain=self.channel_gain.tolist())

    __repr__ = __str__

    @classmethod
    def make_channel_gain(cls, n_tx: int, target_snr_db: Sequence[float],
                          power_alloc: Optional[Sequence[float]] = None,
                          noise_power: float = 1.0,
                          keep_last_noise: bool = False) -> 'AWGNMultiUplinkChannel':
        """
        Fix sigma^2 = noise_power and solve channel_gain

            a_k^2 |h_k|^2 / sigma^2 == s_k     (s_k = 10^(target_snr_db[k]/10))
            |h_k|^2 = s_k * sigma^2 / a_k^2

        power_alloc defaults to a_k == 1 (no NOMA, pure near-far). Equal targets
        reduce to h_k == 1, the homogeneous channel.
        """
        target = torch.as_tensor(target_snr_db, dtype=torch.float32)
        if target.numel() != n_tx:
            raise ValueError(f'{target.numel() = } != {n_tx = }')
        a2 = cls._as_vector(power_alloc, n_tx, default=1.0)

        s = torch.pow(10.0, target / 10.0)
        h2 = s * noise_power / a2.clamp_min(1e-12)

        return cls(n_tx, power_alloc=a2, channel_gain=torch.sqrt(h2),
                  noise_power=noise_power, keep_last_noise=keep_last_noise)
    
    def _effective_gain(self, device, dtype) -> torch.Tensor:
        a = torch.sqrt(self.power_alloc.to(device=device, dtype=dtype))
        h = self.channel_gain.to(device=device, dtype=dtype)
        return a * h


    def _noise_for(self, signal: torch.Tensor) -> torch.Tensor:
        """One user's post-gain signal -> its noise, per this channel's sigma^2 rule."""
        if self.noise_power is not None:
            return make_complex_gaussian_noise(
                signal, torch.as_tensor(self.noise_power, device=signal.device))
        reference_power = signal_power(signal, keepdim=True).detach()
        return make_awgn_noise(signal, self.snr_db, reference_power=reference_power)


    def ofdm(self, signal: torch.Tensor, user_dim_index: int = 0) -> List[torch.Tensor]:
        """
        Args:
            signal: complex tensor of shape (*d1, n_tx, *d2, L)

        Returns:
            list of n_tx complex tensors, each (*d1, *d2, L) -- one stream per user.
        """
        if user_dim_index < 0:
            raise ValueError('user_dim_index must be non-negative')
        if signal.size(user_dim_index) != self.n_tx:
            raise ValueError(f'{signal.size(user_dim_index) = } != {self.n_tx = }')

        view = [1] * signal.dim()
        view[user_dim_index] = self.n_tx
        g = self._effective_gain(signal.device, signal.real.dtype).view(view)
        self.last_effective_gain = g.detach().to('cpu')

        gz = signal * g.to(signal.dtype)
        streams = [zu + self._noise_for(zu) for zu in gz.unbind(dim=user_dim_index)]

        if self.keep_last_noise:
            self.last_noise = [(y - z).detach().to('cpu') for y, z in
                               zip(streams, gz.unbind(dim=user_dim_index))]
        return streams

    def interfere(self, signal: torch.Tensor, user_dim_index: int = 0,
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            signal: complex tensor of shape (*d1, n_tx, *d2, L);
            noise:  optional pre-drawn noise of shape (*d1, *d2, L), for reproducibility.

        Returns:
            complex tensor of shape (*d1, *d2, L) -- one superimposed, noisy stream,
            shared by every receiving decoder.
        """
        if user_dim_index < 0:
            raise ValueError('user_dim_index must be non-negative')
        if signal.size(user_dim_index) != self.n_tx:
            raise ValueError(f'{signal.size(user_dim_index) = } != {self.n_tx = }')

        view = [1] * signal.dim()
        view[user_dim_index] = self.n_tx
        g = self._effective_gain(signal.device, signal.real.dtype).view(view)
        self.last_effective_gain = g.detach().to('cpu')

        signal = signal * g.to(signal.dtype)
        received = torch.sum(signal, dim=user_dim_index)        # (*d1, *d2, L)

        if noise is not None:
            noise = noise.detach().to(received.device)
        else:
            # sigma^2 from the measured power of THIS superimposed signal
            noise = self._noise_for(received)

        if self.keep_last_noise:
            self.last_noise = noise.detach().to('cpu')
        return received + noise

    def get_last_noise(self) -> Optional[torch.Tensor]:
        return self.last_noise

    def get_last_effective_gain(self) -> Optional[torch.Tensor]:
        """a_k * h_k applied by the last interfere()/ofdm() call (CPU), (*d1, n_tx, *d2, 1)."""
        return self.last_effective_gain


class NopChannel(Channel):
    """
    Identity channel: y = x (and the superposition, in the uplink case). Used to sanity
    check the rest of the pipeline -- if the model cannot be trained through a NopChannel,
    the bug is in the codec or the loss, not in the noise.

    Wraps an optional underlying channel so attributes like `snr_db` still resolve, i.e.
    this only overrides interfere().
    """

    def __init__(self, original_channel: Optional[Channel] = None, sum_users: bool = True):
        """
        Args:
            original_channel: the channel being stubbed out, for attribute lookup / logging.
            sum_users: if True, still superimpose when called with a user_dim_index.
                Set False to pass the per-user signals through untouched.
        """
        self.original_channel = original_channel
        self.sum_users = sum_users

    def __str__(self):
        return get_class_str(self, original_channel=self.original_channel)

    __repr__ = __str__

    def ofdm(self, signal: torch.Tensor, user_dim_index: int = 0) -> List[torch.Tensor]:
        """Every user's stream unchanged: no gain, no noise, no interference."""
        return list(signal.unbind(dim=user_dim_index))

    def interfere(self, signal: torch.Tensor, user_dim_index: Optional[int] = None,
                  **kwargs) -> torch.Tensor:
        if user_dim_index is not None and self.sum_users:
            return torch.sum(signal, dim=user_dim_index)
        return signal

    def get_original_channel(self) -> Optional[Channel]:
        return self.original_channel

    def __getattr__(self, name):
        # only called when normal lookup fails
        original = self.__dict__.get('original_channel')
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)


class UniformVariateChannelMixin:
    """
    Mixin that resamples snr_db uniformly in dB before each transmission.

    Usage:
        channel = VariateAWGNMultiUplinkChannel(snr_range=(0, 20), n_tx=2, snr_db=0)
        for batch in loader:
            channel.resample_snr()      # call once per batch, before interfere()
            ...
    """

    def __init__(self, snr_range: Tuple[float, float], *args, **kwargs):
        """
        Args:
            snr_range: (low, high) in dB, sampled uniformly in the dB (log) domain.
            *args, **kwargs: forwarded to the underlying channel class. Its snr_db
                argument can be any value, it is overwritten by resample_snr().
        """
        self.snr_low, self.snr_high = snr_range
        super().__init__(*args, **kwargs)

    def __str__(self):
        return (f'Variate[{super().__str__()}]'
                f'(snr_range=({self.snr_low}, {self.snr_high}), cur_snr_db={self.snr_db})')

    __repr__ = __str__

    def set_snr_range(self, snr_range: Tuple[float, float]):
        self.snr_low, self.snr_high = snr_range

    def resample_snr(self) -> float | torch.Tensor:
        """Draw a new snr_db, used by every interfere() until the next call. Returns it."""
        if isinstance(self.snr_db, torch.Tensor):
            self.snr_db = (torch.rand_like(self.snr_db.float())
                           * (self.snr_high - self.snr_low) + self.snr_low)
        else:
            self.snr_db = torch.rand(1).item() * (self.snr_high - self.snr_low) + self.snr_low
        return self.snr_db


class VariateAWGNSingleChannel(UniformVariateChannelMixin, AWGNSingleChannel):
    pass


class VariateAWGNMultiUplinkChannel(UniformVariateChannelMixin, AWGNMultiUplinkChannel):
    pass


# NOTE on extending to fading:
#   A flat fading channel is  y_r = sum_t h_{t,r} * x_t + n_r,  so it needs exactly two
#   extra pieces: a _make_channel_gain() producing h of shape (*d1, n_tx, n_rx, *d2, L)
#   (Rayleigh: h ~ CN(0, var); slow fading: draw once and expand over L), and an optional
#   divide-by-gain step for perfect channel estimation at the receiver. The AWGN classes
#   above are the h == 1, n_rx == 1 special case. Adding fading later does not change the
#   interfere(signal, user_dim_index) signature the system model depends on, which is why
#   that signature is kept even though AWGN alone would not need user_dim_index.


if __name__ == '__main__':
    torch.manual_seed(0)
    from .utils import power_normalize

    B, L = 64, 4096

    def measured_snr_db(tx: torch.Tensor, rx: torch.Tensor) -> float:
        n = rx - tx
        return (10 * torch.log10(signal_power(tx).mean() / signal_power(n).mean())).item()

    # --- 1. single user: the measured SNR must match the requested one
    for snr in (0.0, 10.0, 20.0):
        x = power_normalize(torch.randn(B, L, dtype=torch.complex64), 1.0)
        y = AWGNSingleChannel(snr).interfere(x)
        got = measured_snr_db(x, y)
        assert abs(got - snr) < 0.1, f'{snr = } {got = }'
        print(f'[ok] single user  requested {snr:5.1f} dB -> measured {got:6.2f} dB')

    # --- 2. power constraint is what makes SNR meaningful:
    #        a 100x louder signal at the same SNR gets 100x louder noise, same result
    loud = power_normalize(torch.randn(B, L, dtype=torch.complex64), 100.0)
    assert abs(measured_snr_db(loud, AWGNSingleChannel(10.0).interfere(loud)) - 10.0) < 0.1
    print('[ok] SNR is invariant to the absolute signal scale')

    # --- 3. uplink: output is the superposition plus noise, a=h=1 by default
    n_tx = 2
    xs = power_normalize(torch.randn(n_tx, B, L, dtype=torch.complex64), 1.0)
    ch = AWGNMultiUplinkChannel(n_tx, snr_db=10.0)
    y = ch.interfere(xs, user_dim_index=0)
    assert y.shape == (B, L), y.shape
    n = y - xs.sum(dim=0)
    got = measured_snr_db(xs.sum(0), y)
    assert abs(got - 10.0) < 0.5, got
    print(f'[ok] uplink shape {tuple(xs.shape)} -> {tuple(y.shape)}, '
          f'measured SNR {got:.2f} dB (requested 10.0)')

    # --- 4. power_alloc and channel_gain both scale the effective per-user gain a_k*h_k
    ch2 = AWGNMultiUplinkChannel(n_tx, snr_db=10.0, power_alloc=[2.0, 0.5],
                                 channel_gain=[1.0, 2.0], keep_last_noise=True)
    ch2.interfere(xs, user_dim_index=0)
    g = ch2.get_last_effective_gain().squeeze()
    assert torch.allclose(g, torch.sqrt(torch.tensor([2.0, 0.5])) * torch.tensor([1.0, 2.0]),
                          atol=1e-4)
    print(f'[ok] effective gain a_k*h_k = {g.tolist()}')

    # --- 5. user dim not at 0, and reproducible noise
    xs2 = torch.randn(B, n_tx, L, dtype=torch.complex64)
    ch2 = AWGNMultiUplinkChannel(n_tx, 10.0, keep_last_noise=True)
    y1 = ch2.interfere(xs2, user_dim_index=1)
    assert y1.shape == (B, L)
    y2 = ch2.interfere(xs2, user_dim_index=1, noise=ch2.get_last_noise())
    assert torch.allclose(y1, y2), 'replaying the stored noise must reproduce the output'
    print('[ok] user_dim_index=1 and noise replay')

    # --- 6. gradients flow to the transmitter, but not into the noise power
    x = torch.randn(B, L, dtype=torch.complex64, requires_grad=True)
    AWGNSingleChannel(10.0).interfere(x).abs().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.abs()).all()
    print('[ok] gradients flow through the channel')

    # --- 7. variate channel
    vc = VariateAWGNMultiUplinkChannel(snr_range=(0, 20), n_tx=2, snr_db=0.0)
    snrs = [vc.resample_snr() for _ in range(1000)]
    assert 0 <= min(snrs) and max(snrs) <= 20
    print(f'[ok] variate SNR in [{min(snrs):.2f}, {max(snrs):.2f}] dB')

    # --- 8. NopChannel still superimposes but adds nothing
    assert torch.allclose(NopChannel().interfere(xs, user_dim_index=0), xs.sum(dim=0))
    assert torch.allclose(NopChannel()(xs, user_dim_index=0), xs.sum(dim=0))  # __call__ dispatch
    print('[ok] NopChannel')

    # --- 9. make_channel_gain: fixed noise_power makes this EXACT, not measured
    want = [15.0, 5.0]
    hc = AWGNMultiUplinkChannel.make_channel_gain(2, want, keep_last_noise=True)
    hc.interfere(xs, user_dim_index=0)
    g = hc.get_last_effective_gain().squeeze()          # (n_tx,)
    got_db = 10 * torch.log10(g.pow(2) / hc.noise_power)
    for t, w in enumerate(want):
        assert abs(got_db[t].item() - w) < 1e-3, f'user {t}: {w = } {got_db[t] = }'
        print(f'[ok] heterogeneous user {t}: requested {w:5.1f} dB -> exactly {got_db[t]:.2f} dB')
    # equal targets and equal power_alloc must give equal channel_gain across users
    # (h == 1 specifically only if noise_power is also chosen to match, since sigma^2
    # is now an independent, fixed constant rather than solved for)
    g_homo = AWGNMultiUplinkChannel.make_channel_gain(2, [10.0, 10.0])
    assert torch.allclose(g_homo.channel_gain[0], g_homo.channel_gain[1], atol=1e-4)
    g_unit = AWGNMultiUplinkChannel.make_channel_gain(2, [10.0, 10.0], noise_power=0.1)
    assert torch.allclose(g_unit.channel_gain, torch.ones(2), atol=1e-4)
    print('[ok] equal targets give equal h; h == 1 when noise_power matches the target')

    # --- 10. direct sanity: interfere() must actually perturb the signal, on every path
    ch_s = AWGNSingleChannel(10.0, keep_last_noise=True)
    y_s = ch_s.interfere(xs[0])
    assert not torch.allclose(xs[0], y_s), 'AWGNSingleChannel produced no change at all'
    assert torch.allclose(y_s, xs[0] + ch_s.get_last_noise())
    print('[ok] AWGNSingleChannel: output = input + noise, and they differ')

    ch_m = AWGNMultiUplinkChannel(n_tx, snr_db=10.0, keep_last_noise=True)
    y_m = ch_m.interfere(xs, user_dim_index=0)
    assert not torch.allclose(xs.sum(dim=0), y_m), 'AWGNMultiUplinkChannel (snr_db) added no noise'
    print('[ok] AWGNMultiUplinkChannel (snr_db path): differs from the noiseless superposition')

    ch_g = AWGNMultiUplinkChannel.make_channel_gain(2, [15.0, 5.0], keep_last_noise=True)
    y_g = ch_g.interfere(xs, user_dim_index=0)
    g = ch_g.get_last_effective_gain().to(xs.device)
    assert not torch.allclose((xs * g).sum(dim=0), y_g), \
        'AWGNMultiUplinkChannel (noise_power path) added no noise'
    print('[ok] AWGNMultiUplinkChannel (noise_power path): differs from the noiseless superposition')

    # --- 11. ofdm(): n_tx independent streams, no interference, each perturbed by noise
    ch_o = AWGNMultiUplinkChannel(n_tx, snr_db=10.0, power_alloc=[2.0, 0.5],
                                  channel_gain=[1.0, 2.0], keep_last_noise=True)
    ys = ch_o.ofdm(xs, user_dim_index=0)
    assert isinstance(ys, list) and len(ys) == n_tx
    g = ch_o.get_last_effective_gain().squeeze()
    for u in range(n_tx):
        assert ys[u].shape == xs[u].shape, (u, ys[u].shape, xs[u].shape)
        assert not torch.allclose(g[u] * xs[u], ys[u]), f'user {u}: ofdm() added no noise'
    # same a_k*h_k as interfere() would use -- only the summation differs
    assert torch.allclose(g, ch_o._effective_gain(xs.device, xs.real.dtype))
    # each user's noise is independent: summing the ofdm streams should NOT reproduce a
    # single-noise-realization superposition (the per-user noise draws don't cancel out
    # the way a shared draw would over many trials, so the two noise tensors differ)
    noise_ofdm = torch.cat([n.reshape(-1) for n in ch_o.get_last_noise()])
    assert noise_ofdm.numel() == n_tx * B * L
    print(f'[ok] ofdm(): {n_tx} independent streams, each perturbed, same gain as interfere()')