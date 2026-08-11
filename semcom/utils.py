"""
Tensor helpers shared by the SemCom modules.

Terminology (kept consistent across the whole `semcom` package):
- "symbol"          : one complex number, i.e. one baseband complex signal symbol.
- "signal"          : a stream of complex symbols, stored in the LAST dimension of a tensor.
                      every dimension before the last one is treated as a batch dimension.
- "power"           : average power PER COMPLEX SYMBOL, i.e. E[|z|^2].
                      note this is per-symbol, not the total energy of the stream.

A neural encoder emits real numbers, but a channel is defined on complex symbols,
so every signal crosses the real<->complex boundary exactly once, via the two
functions below. The convention is: 2 real values <=> 1 complex symbol.
"""
from typing import Literal
import torch

RealComplexMethod = Literal['view', 'concat']


def tensor_real2complex(input: torch.Tensor, method: RealComplexMethod = 'view') -> torch.Tensor:
    """
    Pack a real tensor into a complex one, mapping 2 real values to 1 complex symbol.

    Args:
        input: real (floating point) tensor of shape (*batch, N), N must be even.
        method: how the real values are laid out. Given [a, b, c, d, e, f]:
            - 'view'  : [a+bj, c+dj, e+fj]  (adjacent pairs; this is torch.view_as_complex)
            - 'concat': [a+dj, b+ej, c+fj]  (first half is Re, second half is Im)

    Returns:
        complex tensor of shape (*batch, N // 2)

    Note:
        Differentiable: the result shares storage / graph with `input`, it is not detached.
        Use 'view' unless a paper you are reproducing explicitly says otherwise.
    """
    if input.is_complex():
        raise TypeError('tensor_real2complex() expects a real tensor')
    if input.size(-1) % 2 != 0:
        raise ValueError(f'last dim must be even, got {input.size(-1)}')

    if method == 'view':
        # (*batch, N) -> (*batch, N//2, 2) -> (*batch, N//2) complex
        # .contiguous() is required: view_as_complex needs the last dim to be stride-1
        return torch.view_as_complex(
            input.reshape(*input.size()[:-1], input.size(-1) // 2, 2).contiguous()
        )
    if method == 'concat':
        half = input.size(-1) // 2
        return torch.complex(input[..., :half], input[..., half:])
    raise ValueError(f'invalid method {method!r}, expected "view" or "concat"')


def tensor_complex2real(input: torch.Tensor, method: RealComplexMethod = 'view') -> torch.Tensor:
    """
    Inverse of tensor_real2complex(): unpack 1 complex symbol into 2 real values.

    tensor_complex2real(tensor_real2complex(x, m), m) == x  for both methods.

    Args:
        input: complex tensor of shape (*batch, N)
        method: same convention as tensor_real2complex()

    Returns:
        real tensor of shape (*batch, N * 2)
    """
    if not input.is_complex():
        raise TypeError('tensor_complex2real() expects a complex tensor')

    if method == 'view':
        # interleave: [Re0, Im0, Re1, Im1, ...]
        return torch.stack([input.real, input.imag], dim=-1).reshape(*input.size()[:-1], -1)
    if method == 'concat':
        return torch.cat([input.real, input.imag], dim=-1)
    raise ValueError(f'invalid method {method!r}, expected "view" or "concat"')


def signal_power(signal: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Average power per complex symbol of a signal, measured over the last dimension.

    Args:
        signal: real or complex tensor of shape (*batch, L)
        keepdim: keep the reduced last dimension as size 1 (handy for broadcasting)

    Returns:
        real tensor of shape (*batch, 1) if keepdim else (*batch,)

    Note:
        For a real tensor we still report power PER COMPLEX SYMBOL: L real values are
        L/2 complex symbols, so the power is sum(x^2) / (L/2) == 2 * mean(x^2).
        This is what makes power_normalize() commute with tensor_real2complex().
    """
    if signal.is_complex():
        return signal.abs().pow(2).mean(dim=-1, keepdim=keepdim)
    return 2.0 * signal.pow(2).mean(dim=-1, keepdim=keepdim)


def power_normalize(signal: torch.Tensor,
                    power_constraint: float | torch.Tensor = 1.0,
                    allow_less_power: bool = False,
                    eps: float = 1e-12) -> torch.Tensor:
    """
    Scale each signal in the batch so its average power per complex symbol equals
    `power_constraint`, i.e. after this call  E[|z_k|^2] == power_constraint.

    This is what makes the SNR of a channel meaningful: without it, the encoder could
    trivially "beat" the noise by emitting a huge-amplitude signal.

    Args:
        signal: real or complex tensor of shape (*batch, L). Normalized independently
            per batch element (the reduction is over the last dim only).
        power_constraint: target average power per complex symbol. A float, or a real
            tensor broadcastable to (*batch, 1) for per-user / per-sample constraints.
        allow_less_power: if True, a signal that is already below the constraint is left
            untouched instead of being amplified up to it (an inequality constraint
            E[|z|^2] <= P rather than an equality). Default False.
        eps: guards against dividing by zero for an all-zero signal.

    Returns:
        tensor of the same shape / dtype as `signal`.

    Note:
        power_normalize(x, P) commutes with the real<->complex conversion:
            tensor_real2complex(power_normalize(x, P)) == power_normalize(tensor_real2complex(x), P)
        Gradients flow through the scaling factor, which is intentional: the encoder
        should learn under the constraint, not around it.
    """
    if not isinstance(power_constraint, torch.Tensor):
        power_constraint = torch.tensor(power_constraint, dtype=torch.float32,
                                        device=signal.device)
    power_constraint = power_constraint.to(device=signal.device, dtype=torch.float32)

    power = signal_power(signal, keepdim=True).clamp_min(eps)

    if allow_less_power:
        # scale < 1 only; leave already-quiet signals alone
        power = torch.maximum(power, power_constraint)

    scale = torch.sqrt(power_constraint / power)
    if signal.is_complex():
        scale = scale.to(signal.real.dtype)
    return signal * scale


def get_class_str(obj, **kwargs) -> str:
    """Pretty __str__ helper: `semcom.channel.AWGNSingleChannel(snr_db=10)`."""
    module = type(obj).__module__
    name = type(obj).__qualname__
    args = ', '.join(f'{k}={v}' for k, v in kwargs.items())
    return f'{module}.{name}({args})'


if __name__ == '__main__':
    torch.manual_seed(0)

    # round trip both ways, both methods
    for method in ('view', 'concat'):
        x = torch.randn(4, 8)
        assert torch.allclose(tensor_complex2real(tensor_real2complex(x, method), method), x)
        z = torch.randn(4, 4, dtype=torch.complex64)
        assert torch.allclose(tensor_real2complex(tensor_complex2real(z, method), method), z)
    print('[ok] real <-> complex round trip')

    # power normalize: complex
    z = torch.randn(16, 1024, dtype=torch.complex64) * 7.5
    zn = power_normalize(z, 1.0)
    assert torch.allclose(signal_power(zn), torch.ones(16, 1), atol=1e-5)
    # power normalize: real, and it commutes with the packing
    x = torch.randn(16, 2048) * 7.5
    assert torch.allclose(tensor_real2complex(power_normalize(x, 2.0)),
                          power_normalize(tensor_real2complex(x), 2.0), atol=1e-5)
    # allow_less_power leaves a quiet signal alone
    quiet = torch.randn(16, 1024, dtype=torch.complex64) * 0.01
    assert torch.allclose(power_normalize(quiet, 1.0, allow_less_power=True), quiet)
    print('[ok] power_normalize')
