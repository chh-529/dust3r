"""
The semantic codec: DUSt3R ViT features <-> channel symbols.

This is the DeepSC channel encoder / decoder (vendored verbatim in semcom/deepsc.py)
wrapped with the shape and power handling this project needs. Nothing about the
compression itself is invented here -- the two nn.Sequential stacks are DeepSC's.

Where this sits in the whole system:

    image -> [DUSt3R ViT encoder] -> feat (B, V, S, D)     <- "semantic encoder"
          -> [FeatureChannelEncoder] -> z (B, L) complex   <- transmitted
          -> [channel] -> y (B, L) complex
          -> [FeatureChannelDecoder] -> feat_hat (B, V, S, D)
          -> [DUSt3R decoder + heads] -> pointmaps          <- "semantic decoder"

So DUSt3R plays the role of DeepSC's semantic encoder / decoder, and this file is
only the channel encoder / decoder in between. See semcom/deepsc.py for why.

Bandwidth
---------
Following DeepJSCC, the bandwidth ratio is (complex channel symbols) / (source real
dimensions). There are two useful choices of "source" here, and they differ only by a
constant, so we report both -- see bandwidth_report():

    rho_feat = k / D                 how hard the channel encoder squeezes a token
                                     (DeepSC's channel encoder runs at 8/128 = 1/16)
    rho_img  = k * S / (3 * H * W)   the DeepJSCC number, comparable to their 1/6, 1/12

with k the complex symbols emitted per token. Note k / D = 1/2 is the point where the
transmitted real degrees of freedom equal the feature's own (1 complex symbol == 2
reals); beyond that you are spending bandwidth on redundancy rather than compression.
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .deepsc import ChannelDecoder, ChannelEncoder
from .utils import power_normalize, tensor_complex2real, tensor_real2complex

# DUSt3R_ViTLarge_BaseDecoder_512dpt at 512x384 input
DEFAULT_FEAT_DIM = 1024          # enc_embed_dim
DEFAULT_TOKENS_PER_IMAGE = 768   # (512/16) * (384/16)
DEFAULT_IMAGE_SHAPE = (3, 384, 512)


def symbols_per_token(feat_dim: int, bandwidth_ratio: float) -> int:
    """
    k = round(D * rho). Rounded because D is rarely divisible by the ratio you want
    (1024 / 3 is not an integer), so the realized ratio can differ from the nominal one
    by up to ~0.5%; bandwidth_report() prints the realized value.
    """
    k = int(round(feat_dim * bandwidth_ratio))
    if k < 1:
        raise ValueError(f'{bandwidth_ratio = } is too small for {feat_dim = }')
    return k


def bandwidth_report(feat_dim: int = DEFAULT_FEAT_DIM,
                     tokens_per_image: int = DEFAULT_TOKENS_PER_IMAGE,
                     image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
                     views_per_user: int = 4,
                     ratios: Sequence[float] = (1 / 2, 1 / 3, 1 / 6, 1 / 12, 1 / 16)) -> str:
    """Table of what each nominal bandwidth ratio actually costs. See module docstring."""
    n_img = image_shape[0] * image_shape[1] * image_shape[2]
    lines = [
        f'feature {tokens_per_image} tokens x {feat_dim} dim = '
        f'{tokens_per_image * feat_dim:,} reals/image',
        f'image   {image_shape[0]}x{image_shape[1]}x{image_shape[2]} = {n_img:,} reals/image'
        f'   (the feature is {tokens_per_image * feat_dim / n_img:.2f}x the image)',
        '',
        f'{"rho_feat":>10} | {"k":>5} | {"C=2k":>5} | {"realized":>9} | '
        f'{"rho_img":>8} | {"symbols/user":>13}',
        '-' * 68,
    ]
    for r in ratios:
        k = symbols_per_token(feat_dim, r)
        lines.append(
            f'{r:>10.4f} | {k:>5} | {2 * k:>5} | {k / feat_dim:>9.4f} | '
            f'{k * tokens_per_image / n_img:>8.4f} | '
            f'{k * tokens_per_image * views_per_user:>13,}'
        )
    return '\n'.join(lines)


class FeatureChannelEncoder(nn.Module):
    """
    Transmitter side: ViT features -> power-constrained complex symbols.

    Follows DeepSC's DeepSC_EDP.make_signal() step for step:
        channel_encoder -> flatten(start_dim=1) -> real2complex('view') -> power normalize
    (references/semcom/paper_src/model/edp/deepsc.py). The only addition is the explicit
    power constraint, which DeepSC applies in its own power_normalize.
    """

    def __init__(self,
                 feat_dim: int = DEFAULT_FEAT_DIM,
                 bandwidth_ratio: float = 1 / 6,
                 hidden_dim: Optional[int] = None,
                 power_constraint: float = 1.0):
        """
        Args:
            feat_dim: D, the ViT encoder embedding dim.
            bandwidth_ratio: rho_feat = k / feat_dim, complex symbols per token.
            hidden_dim: width of the channel encoder's middle layer. Defaults to 2*D,
                matching DeepSC's ratio (d_model=128 -> 256).
            power_constraint: average power per complex symbol, E[|z|^2].
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.bandwidth_ratio = bandwidth_ratio
        self.symbols_per_token = symbols_per_token(feat_dim, bandwidth_ratio)
        self.reals_per_token = 2 * self.symbols_per_token
        self.power_constraint = power_constraint

        hidden_dim = hidden_dim if hidden_dim is not None else 2 * feat_dim
        self.channel_encoder = ChannelEncoder(feat_dim, hidden_dim, self.reals_per_token)

    def extra_repr(self) -> str:
        return (f'feat_dim={self.feat_dim}, rho_feat={self.bandwidth_ratio:.4f}, '
                f'k={self.symbols_per_token}, power={self.power_constraint}')

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: real tensor (B, *token_dims, D). token_dims is typically (V, S) for V
                views of S patch tokens, but any number of dims works -- they are all
                flattened into the symbol stream.

        Returns:
            complex tensor (B, L) with L = prod(token_dims) * k, power normalized so
            E[|z|^2] == power_constraint per batch element.
        """
        x = self.channel_encoder(feat)          # (B, *token_dims, 2k)
        # view_as_complex needs float32/64; under autocast the Linear above returns fp16
        x = x.float().flatten(start_dim=1)      # (B, prod(token_dims) * 2k)
        z = tensor_real2complex(x, 'view')      # (B, L)
        return power_normalize(z, self.power_constraint)


class FeatureChannelDecoder(nn.Module):
    """
    Receiver side: noisy complex symbols -> ViT feature space.

    Mirrors DeepSC's DeepSC_EDP.decode_from_signal():
        complex2real('view') -> reshape back to tokens -> channel_decoder

    DeepSC's ChannelDecoder ends in a LayerNorm, which happens to line up with DUSt3R:
    its ViT features come out of enc_norm (also a LayerNorm), so feat_hat lands in the
    same scale the DUSt3R decoder expects and needs no extra rescaling.
    """

    def __init__(self,
                 feat_dim: int = DEFAULT_FEAT_DIM,
                 bandwidth_ratio: float = 1 / 6,
                 inner_dim: Optional[int] = None):
        """
        Args:
            feat_dim: D, must match the encoder.
            bandwidth_ratio: must match the encoder.
            inner_dim: width of the channel decoder's inner expansion. Defaults to 4*D,
                matching DeepSC's ratio (size1=128 -> size2=512).
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.bandwidth_ratio = bandwidth_ratio
        self.symbols_per_token = symbols_per_token(feat_dim, bandwidth_ratio)
        self.reals_per_token = 2 * self.symbols_per_token

        inner_dim = inner_dim if inner_dim is not None else 4 * feat_dim
        self.channel_decoder = ChannelDecoder(self.reals_per_token, feat_dim, inner_dim)

    def extra_repr(self) -> str:
        return (f'feat_dim={self.feat_dim}, rho_feat={self.bandwidth_ratio:.4f}, '
                f'k={self.symbols_per_token}')

    def forward(self, z: torch.Tensor, token_dims: Sequence[int]) -> torch.Tensor:
        """
        Args:
            z: complex tensor (B, L), the received signal.
            token_dims: the token layout the encoder was given, e.g. (V, S). Passed in
                rather than remembered, so the decoder stays a pure function of what it
                receives -- the receiver knows the image geometry as side information,
                it is not carried over the channel.

        Returns:
            real tensor (B, *token_dims, D)
        """
        x = tensor_complex2real(z, 'view')                       # (B, L*2)
        x = x.reshape(x.size(0), *token_dims, self.reals_per_token)
        return self.channel_decoder(x)


def make_codec_pair(feat_dim: int = DEFAULT_FEAT_DIM,
                    bandwidth_ratio: float = 1 / 6,
                    power_constraint: float = 1.0,
                    hidden_dim: Optional[int] = None,
                    inner_dim: Optional[int] = None
                    ) -> Tuple[FeatureChannelEncoder, FeatureChannelDecoder]:
    """
    Build a matched encoder/decoder. Use one call per user to get INDEPENDENT weights,
    or reuse one pair for every user to share weights.

    Sharing is fine for a first single-transmitter experiment. Once two users
    superimpose, sharing the encoder makes the map (f1, f2) -> z1 + z2 symmetric under
    swapping the users, so no decoder can tell which user sent what; break the symmetry
    with independent weights or a small per-user signature added to the code.
    """
    enc = FeatureChannelEncoder(feat_dim, bandwidth_ratio, hidden_dim, power_constraint)
    dec = FeatureChannelDecoder(feat_dim, bandwidth_ratio, inner_dim)
    return enc, dec


if __name__ == '__main__':
    from .channel import AWGNSingleChannel, NopChannel
    from .utils import signal_power

    torch.manual_seed(0)

    print(bandwidth_report())
    print()

    B, V, S, D = 2, 4, DEFAULT_TOKENS_PER_IMAGE, DEFAULT_FEAT_DIM
    rho = 1 / 6
    enc, dec = make_codec_pair(D, rho)
    print(enc)
    print(dec)

    n_enc = sum(p.numel() for p in enc.parameters())
    n_dec = sum(p.numel() for p in dec.parameters())
    print(f'\nparameters: encoder {n_enc:,}  decoder {n_dec:,}  total {n_enc + n_dec:,}')

    # DUSt3R features come out of a LayerNorm, so unit-ish scale is the realistic input
    feat = torch.randn(B, V, S, D)

    z = enc(feat)
    expected_L = V * S * enc.symbols_per_token
    assert z.shape == (B, expected_L), (z.shape, expected_L)
    assert z.is_complex()
    print(f'\n[ok] encode {tuple(feat.shape)} -> {tuple(z.shape)} complex '
          f'(L = {V}*{S}*{enc.symbols_per_token} = {expected_L:,})')

    assert torch.allclose(signal_power(z), torch.ones(B, 1), atol=1e-4)
    print(f'[ok] power constraint satisfied: E[|z|^2] = {signal_power(z).mean():.6f}')

    for name, ch in (('noiseless', NopChannel()), ('10 dB AWGN', AWGNSingleChannel(10.0))):
        feat_hat = dec(ch.interfere(z), token_dims=(V, S))
        assert feat_hat.shape == feat.shape, feat_hat.shape
        print(f'[ok] {name:>11}: decode -> {tuple(feat_hat.shape)}, '
              f'output std {feat_hat.std().item():.4f} (LayerNorm-ed, so ~1)')

    # gradients must reach the encoder through the complex channel
    feat = torch.randn(B, V, S, D, requires_grad=True)
    AWGNSingleChannel(10.0).interfere(enc(feat)).abs().sum().backward()
    first_layer = enc.channel_encoder.model[0]
    assert first_layer.weight.grad is not None
    assert torch.isfinite(first_layer.weight.grad).all()
    print('[ok] gradients reach the channel encoder through the channel')

    # a 3-D input (no view dim) must work too: (B, S, D)
    z2 = enc(torch.randn(B, S, D))
    assert z2.shape == (B, S * enc.symbols_per_token)
    assert dec(z2, token_dims=(S,)).shape == (B, S, D)
    print('[ok] works without a view dimension as well')
