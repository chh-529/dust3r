"""
Channel models for SemCom. AWGN only (see NOTE at the bottom on extending to fading).

Conventions
-----------
A signal tensor always keeps the complex symbol stream in its LAST dimension.
Everything before that is a batch dimension, except for one distinguished
"user dimension" in the multi-user case, whose position is passed explicitly as
`user_dim_index` (it is *not* assumed to be dim 0, because the caller may want
(batch, n_tx, L) or (n_tx, batch, L) depending on how the system model stacks things).

Single user (point to point):
    y = x + n                       shape (*batch, L) -> (*batch, L)

Multi user uplink (superposition / non-orthogonal multiple access):
    y = sum_t h_t * x_t + n         shape (*d1, n_tx, *d2, L) -> (*d1, *d2, L)
    All transmitters occupy the SAME L symbols at the same time. The receiver gets
    one superimposed stream and the semantic decoders must pull the users apart --
    that separation is learned, there is no explicit multi-user detector here.

    There is one receiver, therefore one antenna and ONE noise realization: users do
    not get "their own noise". Different link qualities per user (the near-far effect)
    are modelled by the per-user gain h_t instead; pass a per-user snr_db list and the
    channel solves for the h_t that realize them.

Noise model
-----------
n ~ CN(0, sigma^2), circularly symmetric complex Gaussian: the real and the
imaginary part are independent N(0, sigma^2 / 2), so E[|n|^2] == sigma^2.
sigma^2 is derived from the SNR:  sigma^2 = P_ref / 10^(snr_db / 10).

What P_ref is, is a real modelling choice once more than one user transmits --
see `snr_reference` in AWGNMultiUplinkChannel.
"""
from typing import Literal, Optional, Sequence, Tuple

import torch

from .utils import get_class_str, signal_power

SNRReference = Literal['received', 'per_user']


def make_complex_gaussian_noise(signal: torch.Tensor,
                                noise_power: torch.Tensor) -> torch.Tensor:
    """
    Draw n ~ CN(0, noise_power), circularly symmetric complex Gaussian.

    Args:
        signal: complex tensor; only its shape / device / dtype are used.
        noise_power: sigma^2, a real tensor broadcastable to `signal`'s shape.

    Returns:
        complex noise tensor shaped like `signal`, with E[|n|^2] == noise_power.
    """
    # Re and Im each carry half the power -> E[|n|^2] == sigma^2
    std = torch.sqrt(noise_power / 2.0)
    noise_real = torch.randn_like(signal.real) * std
    noise_imag = torch.randn_like(signal.imag) * std
    return torch.complex(noise_real, noise_imag)


def make_awgn_noise(signal: torch.Tensor,
                    snr_db: float | torch.Tensor,
                    reference_power: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Draw circularly symmetric complex Gaussian noise for `signal`, without adding it.

    Args:
        signal: complex tensor of any shape; only its shape / device / dtype are used
            (plus its power, if `reference_power` is not given).
        snr_db: SNR in dB. A float, or a real tensor broadcastable to `signal`'s shape
            (per-sample or per-receiver SNR).
        reference_power: the signal power the SNR is measured against, a real tensor
            broadcastable to `signal`'s shape. If None, it is measured from `signal`
            itself, over the last dimension.

    Returns:
        complex noise tensor with the same shape / dtype / device as `signal`.

    Note:
        The reference power is .detach()ed. The noise variance must be treated as a
        constant of the environment; if gradients flowed through it, the encoder could
        reduce its own loss by shaping the noise, which is not physical.
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
    Superposition uplink:  y = sum_{t=1..n_tx} h_t * x_t + n,   n ~ CN(0, sigma^2)

    This is the channel in the architecture diagram: transmitter 1 and transmitter 2
    send z1 and z2 over the same resource, the receiver observes z1' + z2'.

    There is exactly ONE receiver, hence ONE antenna, ONE RF chain and ONE noise
    realization -- sigma^2 is a property of the receiver hardware, not of a user.
    Users therefore cannot have "their own noise". Heterogeneous link quality is
    modelled where it physically lives: in the per-user gain h_t (path loss /
    shadowing), which makes the users arrive at the receiver with different powers:

        SNR_t = |h_t|^2 * P_t / sigma^2

    Homogeneous case (scalar snr_db): h_t == 1 for every user.
    Heterogeneous case (snr_db is a length-n_tx sequence): h_t is solved for, so that
    each user hits its requested received SNR. See _solve_user_gain().
    """

    def __init__(self,
                 n_tx: int,
                 snr_db: float | Sequence[float] | torch.Tensor,
                 snr_reference: SNRReference = 'per_user',
                 keep_last_noise: bool = False):
        """
        Args:
            n_tx: number of transmitters (users) that superimpose.
            snr_db: SNR in dB at the single receiver.
                - a float: every user arrives with the same SNR (h_t == 1).
                - a length-n_tx sequence / 1-D tensor: the RECEIVED SNR of each user,
                  i.e. the near-far / heterogeneous-channel setting. The channel then
                  applies a per-user gain to realize them; `snr_reference` is forced to
                  'per_user' because each user's SNR is stated explicitly.
            snr_reference: WHICH power the SNR is measured against. This matters as soon
                as n_tx > 1, and it decides how your SNR curves should be read:

                'per_user' (default): sigma^2 = mean_t E[|x_t|^2] / snr_linear.
                    "10 dB" means each user arrives 10 dB above the noise. Adding a second
                    user then does NOT change the noise floor, so a 1-user run and a
                    2-user run at the same nominal SNR are directly comparable and the
                    only thing that changed is the mutual interference. This is what you
                    want for the "superposition vs. orthogonal" comparison.

                'received': sigma^2 = E[|sum_t x_t|^2] / snr_linear, i.e. measured on the
                    superimposed waveform (what a receiver's AGC would actually see).
                    With n_tx users this makes the noise ~n_tx times stronger than the
                    'per_user' setting at the same nominal SNR. This matches the
                    reference implementation in references/semcom/paper_src/channel.py.

            keep_last_noise: see AWGNSingleChannel.
        """
        if n_tx < 1:
            raise ValueError(f'{n_tx = } must be >= 1')
        if snr_reference not in ('per_user', 'received'):
            raise ValueError(f'invalid {snr_reference = }')

        if isinstance(snr_db, (list, tuple)):
            snr_db = torch.tensor(snr_db, dtype=torch.float32)
        # a 1-D tensor of length n_tx means "one SNR per user"; anything else is a scalar
        self.per_user_snr = isinstance(snr_db, torch.Tensor) and snr_db.numel() == n_tx > 1
        if self.per_user_snr:
            if snr_db.dim() != 1:
                raise ValueError(f'per-user snr_db must be 1-D, got {snr_db.size()}')
            if snr_reference == 'received':
                raise ValueError("snr_reference='received' is meaningless with per-user "
                                 "snr_db: each user's SNR is already stated explicitly")

        self.n_tx = n_tx
        self.snr_db = snr_db
        self.snr_reference = snr_reference
        self.keep_last_noise = keep_last_noise
        self.last_noise: Optional[torch.Tensor] = None
        self.last_user_gain: Optional[torch.Tensor] = None

    def __str__(self):
        return get_class_str(self, n_tx=self.n_tx, snr_db=self.snr_db,
                             snr_reference=self.snr_reference)

    __repr__ = __str__

    def _solve_user_gain(self, signal: torch.Tensor,
                         user_dim_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Heterogeneous case: find the per-user gains h_t and the noise power sigma^2 that
        realize the requested per-user received SNRs.

        Let p_t = E[|x_t|^2] be what user t transmits (measured from the signal), and
        s_t = 10^(snr_db[t] / 10) the requested linear SNR. We need the received power
        q_t = |h_t|^2 * p_t to satisfy q_t / sigma^2 == s_t. That is n_tx equations in
        n_tx + 1 unknowns, so one more condition pins the absolute scale; we keep the
        TOTAL received power unchanged, sum_t q_t == sum_t p_t:

            sigma^2   = mean_t(p_t) / mean_t(s_t)
            |h_t|^2   = s_t * sigma^2 / p_t

        Only the ratios q_t : sigma^2 affect the system (they are what SNR and SINR are
        made of), so this extra condition is free; keeping the received power stable
        across configurations just makes training better behaved.

        Sanity check -- if every s_t == s and every p_t == p, this gives sigma^2 = p / s
        and h_t == 1, i.e. exactly the homogeneous channel.

        Returns:
            gain: real tensor of shape (*d1, n_tx, *d2, 1), broadcasting over the symbols
            noise_power: real tensor of shape (*d1, *d2, 1)

        Note:
            The measured power is detached: the gain is a property of the environment,
            not something the encoder should be able to steer via its own amplitude.
            In practice the system model power_normalize()s before transmitting, so
            p_t is just the power constraint P_t.
        """
        # (*d1, n_tx, *d2, 1)
        p = signal_power(signal, keepdim=True).detach()

        # reshape snr to (1, ..., n_tx, ..., 1) so it lines up with the user dimension
        view = [1] * signal.dim()
        view[user_dim_index] = self.n_tx
        s = torch.pow(10.0, self.snr_db.to(signal.device, torch.float32) / 10.0).view(view)

        noise_power = p.mean(dim=user_dim_index, keepdim=True) / s.mean()
        gain = torch.sqrt(s * noise_power / p.clamp_min(1e-12))

        return gain, noise_power.squeeze(user_dim_index)

    def interfere(self, signal: torch.Tensor, user_dim_index: int = 0,
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            signal: complex tensor of shape (*d1, n_tx, *d2, L); `user_dim_index` is the
                index of the n_tx dimension (may be negative-free only: use a
                non-negative index, it is also used to place the reduced dims).
            user_dim_index: position of the user dimension. Defaults to 0.
            noise: optional pre-drawn noise of shape (*d1, *d2, L), for reproducibility.

        Returns:
            complex tensor of shape (*d1, *d2, L) -- one superimposed, noisy stream,
            shared by every receiving decoder.
        """
        if user_dim_index < 0:
            raise ValueError('user_dim_index must be non-negative')
        if signal.size(user_dim_index) != self.n_tx:
            raise ValueError(f'{signal.size(user_dim_index) = } != {self.n_tx = }')

        noise_power = None
        if self.per_user_snr:
            # heterogeneous links: attenuate/amplify each user, then one shared noise
            gain, noise_power = self._solve_user_gain(signal, user_dim_index)
            self.last_user_gain = gain.detach().to('cpu')
            signal = signal * gain
            reference_power = None
        elif self.snr_reference == 'per_user':
            # reference power is computed BEFORE summing, so 'per_user' is exact rather
            # than relying on E|sum x_t|^2 == sum E|x_t|^2 (true only for uncorrelated users)
            # (*d1, n_tx, *d2, 1) -> (*d1, *d2, 1)
            reference_power = signal_power(signal, keepdim=True).mean(dim=user_dim_index)
        else:
            reference_power = None

        # the superposition itself: this single line IS the multiple access channel
        received = torch.sum(signal, dim=user_dim_index)        # (*d1, *d2, L)

        if noise is not None:
            noise = noise.detach().to(received.device)
        elif noise_power is not None:
            noise = make_complex_gaussian_noise(received, noise_power)
        else:
            noise = make_awgn_noise(received, self.snr_db, reference_power=reference_power)

        if self.keep_last_noise:
            self.last_noise = noise.detach().to('cpu')
        return received + noise

    def get_last_noise(self) -> Optional[torch.Tensor]:
        return self.last_noise

    def get_last_user_gain(self) -> Optional[torch.Tensor]:
        """
        The per-user gain |h_t| applied by the last interfere() call (CPU), shape
        (*d1, n_tx, *d2, 1). None in the homogeneous case, where h_t == 1.
        """
        return self.last_user_gain


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

    Rationale: training at one fixed SNR gives a model that is only good at that SNR,
    so every operating point needs its own checkpoint. Randomizing the SNR each batch
    yields a single model that degrades gracefully across the whole range, which is
    what you want for the SNR-vs-3D-error curve.

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

    # --- 3. uplink: output is the superposition plus noise
    n_tx = 2
    xs = power_normalize(torch.randn(n_tx, B, L, dtype=torch.complex64), 1.0)
    ch = AWGNMultiUplinkChannel(n_tx, snr_db=10.0, snr_reference='per_user')
    y = ch.interfere(xs, user_dim_index=0)
    assert y.shape == (B, L), y.shape
    # noise power should be per-user power / 10 == 0.1
    n = y - xs.sum(dim=0)
    print(f'[ok] uplink shape {tuple(xs.shape)} -> {tuple(y.shape)}, '
          f'noise power {signal_power(n).mean().item():.4f} (expect 0.1000)')

    # --- 4. the two snr_reference conventions differ by exactly n_tx
    n_recv = AWGNMultiUplinkChannel(n_tx, 10.0, 'received').interfere(xs) - xs.sum(0)
    ratio = (signal_power(n_recv).mean() / signal_power(n).mean()).item()
    print(f'[ok] received/per_user noise power ratio = {ratio:.3f} (expect ~{n_tx})')

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

    # --- 9. heterogeneous per-user SNR: each user must hit its own requested SNR
    want = [15.0, 5.0]
    hc = AWGNMultiUplinkChannel(2, want, keep_last_noise=True)
    y = hc.interfere(xs, user_dim_index=0)
    g = hc.get_last_user_gain()                    # (n_tx, B, 1)
    sigma2 = signal_power(hc.get_last_noise()).mean()
    for t, w in enumerate(want):
        q = signal_power(g[t] ** 2 * xs[t]).mean()  # received power of user t
        got = (10 * torch.log10(q / sigma2)).item()
        assert abs(got - w) < 0.1, f'user {t}: {w = } {got = }'
        print(f'[ok] heterogeneous user {t}: requested {w:5.1f} dB -> received {got:6.2f} dB '
              f'(|h| = {g[t].mean().item():.3f})')
    # total received power is preserved, and the homogeneous case is the special case
    assert abs(signal_power((g * xs).sum(0)).mean().item()
               - signal_power(xs.sum(0)).mean().item()) < 0.05
    # equal per-user SNRs must reduce to the homogeneous channel: h_t == 1
    g_homo = AWGNMultiUplinkChannel(2, [10.0, 10.0])
    g_homo.interfere(xs)
    assert torch.allclose(g_homo.get_last_user_gain(), torch.ones(1), atol=1e-4)
    print('[ok] heterogeneous: total power preserved, equal SNRs reduce to h == 1')
