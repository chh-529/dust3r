"""
The semantic codec: DUSt3R ViT features <-> channel symbols.

This is the DeepSC channel encoder / decoder

Where this sits in the whole system:

    image -> [DUSt3R ViT encoder]       -> feat (B, V, S, D)
          -> [FeatureChannelEncoder]    -> z (B, L) complex
          -> [channel]                  -> y (B, L) complex
          -> [FeatureChannelDecoder]    -> feat_hat (B, V, S, D)
          -> [DUSt3R decoder + heads]   -> pointmaps

So DUSt3R plays the role of DeepSC's semantic encoder / decoder, and this file is
only the channel encoder / decoder in between.

Bandwidth
---------

    cpr = k / D                 compression rate
    bdr = k * S / (3 * H * W)   bandwidth ratio

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


def symbols_per_token(feat_dim: int, cpr: float) -> int:
    """
    k = round(D * cpr).
    """
    k = int(round(feat_dim * cpr))
    if k < 1:
        raise ValueError(f'{cpr = } is too small for {feat_dim = }')
    return k


class FeatureChannelEncoder(nn.Module):
    """
    Transmitter side: ViT features -> power-constrained complex symbols.
    Follows DeepSC's DeepSC_EDP.make_signal() step for step:
        channel_encoder -> flatten(start_dim=1) -> real2complex('view') -> power normalize
    """

    def __init__(self,
                 feat_dim: int = DEFAULT_FEAT_DIM,
                 cpr: float = 1 / 6,
                 hidden_dim: Optional[int] = None,
                 power_constraint: float = 1.0):
        """
        Args:
            feat_dim:           D, the ViT encoder embedding dim.
            cpr:                k / D, complex symbols per token.
            hidden_dim:         width of the channel encoder's middle layer. Defaults to 2*D,
                                matching DeepSC's ratio (d_model=128 -> 256).
            power_constraint:   average power per complex symbol, E[|z|^2].
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.cpr = cpr
        self.symbols_per_token = symbols_per_token(feat_dim, cpr)
        self.reals_per_token = 2 * self.symbols_per_token
        self.power_constraint = power_constraint

        hidden_dim = hidden_dim if hidden_dim is not None else 2 * feat_dim
        self.channel_encoder = ChannelEncoder(feat_dim, hidden_dim, self.reals_per_token)

    def extra_repr(self) -> str:
        return (f'feat_dim={self.feat_dim}, cpr={self.cpr:.4f}, '
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
                 cpr: float = 1 / 6,
                 inner_dim: Optional[int] = None):
        """
        Args:
            feat_dim:           D, must match the encoder.
            cpr:                k / feat_dim, complex symbols per token.
            inner_dim:          width of the channel decoder's inner expansion. Defaults to 4*D,
                                matching DeepSC's ratio (size1=128 -> size2=512).
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.cpr = cpr
        self.symbols_per_token = symbols_per_token(feat_dim, cpr)
        self.reals_per_token = 2 * self.symbols_per_token

        inner_dim = inner_dim if inner_dim is not None else 4 * feat_dim
        self.channel_decoder = ChannelDecoder(self.reals_per_token, feat_dim, inner_dim)

    def extra_repr(self) -> str:
        return (f'feat_dim={self.feat_dim}, cpr={self.cpr:.4f}, '
                f'k={self.symbols_per_token}')

    def forward(self, z: torch.Tensor, token_dims: Sequence[int]) -> torch.Tensor:
        """
        Args:
            z: complex tensor (B, L), the received signal.
            token_dims: the token layout the encoder was given, e.g. (V, S).

        Returns:
            real tensor (B, *token_dims, D)
        """
        x = tensor_complex2real(z, 'view')                       # (B, L*2)
        x = x.reshape(x.size(0), *token_dims, self.reals_per_token)
        return self.channel_decoder(x)


def make_codec_pair(feat_dim: int = DEFAULT_FEAT_DIM,
                    cpr: float = 1 / 6,
                    power_constraint: float = 1.0,
                    hidden_dim: Optional[int] = None,
                    inner_dim: Optional[int] = None
                    ) -> Tuple[FeatureChannelEncoder, FeatureChannelDecoder]:
    """
    Make a matching encoder / decoder pair with the same feature dim and compression rate.
    """
    enc = FeatureChannelEncoder(feat_dim, cpr, hidden_dim, power_constraint)
    dec = FeatureChannelDecoder(feat_dim, cpr, inner_dim)
    return enc, dec


if __name__ == '__main__':
    from .channel import AWGNSingleChannel, NopChannel
    from .utils import signal_power

    torch.manual_seed(0)

    B, V, S, D = 2, 4, DEFAULT_TOKENS_PER_IMAGE, DEFAULT_FEAT_DIM
    cpr = 1 / 6
    enc, dec = make_codec_pair(D, cpr)
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
