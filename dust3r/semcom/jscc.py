"""
Phase B JSCC: Encoder and Decoder modules for Semantic Communication.

Two variants are provided:

LinearJSCCEncoder / LinearJSCCDecoder
    Single linear projection D→k / k→D.
    Equivalent to a linear autoencoder.  Optimal for Gaussian sources + AWGN
    (Shannon-optimal in that case), but cannot exploit non-linear structure in
    ViT features.

JSCCEncoder / JSCCDecoder  (recommended — "real" DeepJSCC)
    Two-layer MLP with GELU non-linearity and LayerNorm, following the
    DeepJSCC architecture (Bourtsoulatze et al. 2019) adapted for semantic
    features (Xie et al. DEEPSC 2021).

    Encoder: LayerNorm(D) → Linear(D,H) → GELU → Linear(H,k)
    Decoder: Linear(k,H) → GELU → Linear(H,D) → LayerNorm(D)

    The non-linearity allows the model to exploit complex statistical structure
    of ViT features that linear projections cannot capture.

The compression ratio is k / D in both cases.  Power normalisation is applied
*outside* these modules (in PhaseBSemCom) so the channel power constraint is
enforced in a unified place.
"""

import math

import torch
import torch.nn as nn


class LinearJSCCEncoder(nn.Module):
    """
    Linear JSCC Encoder: feat (B, N, D) → z (B, N, k).

    A single linear projection from feature dimension D to channel dimension k.
    Power normalisation is applied *outside* this module (in PhaseBSemCom) so
    that the channel power constraint is enforced in a unified place.

    Args:
        feat_dim    : Input feature dimension D (e.g. 1024 for ViT-Large).
        channel_dim : Channel symbol dimension k.
    """

    def __init__(self, feat_dim: int, channel_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.channel_dim = channel_dim
        self.proj = nn.Linear(feat_dim, channel_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat : (B, N, D)
        Returns:
            z    : (B, N, k)
        """
        return self.proj(feat)

    @property
    def compression_ratio(self) -> float:
        """k / D — fraction of the original feature dimension retained."""
        return self.channel_dim / self.feat_dim

    def extra_repr(self) -> str:
        return (
            f'feat_dim={self.feat_dim}, channel_dim={self.channel_dim}, '
            f'ratio={self.compression_ratio:.4f}'
        )


class LinearJSCCDecoder(nn.Module):
    """
    Linear JSCC Decoder: z_hat (B, N, k) → feat_hat (B, N, D).

    A single linear projection from channel dimension k back to feature
    dimension D.  The decoder sees power-normalised received symbols; it must
    learn to both de-compress and de-normalise.

    Args:
        channel_dim : Channel symbol dimension k.
        feat_dim    : Output feature dimension D (e.g. 1024 for ViT-Large).
    """

    def __init__(self, channel_dim: int, feat_dim: int):
        super().__init__()
        self.channel_dim = channel_dim
        self.feat_dim = feat_dim
        self.proj = nn.Linear(channel_dim, feat_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat    : (B, N, k)  received channel symbols (unit-power scale)
        Returns:
            feat_hat : (B, N, D)  reconstructed features
        """
        return self.proj(z_hat)

    def extra_repr(self) -> str:
        return f'channel_dim={self.channel_dim}, feat_dim={self.feat_dim}'


# ── Non-linear DeepJSCC (recommended) ────────────────────────────────────────

class JSCCEncoder(nn.Module):
    """
    Non-linear JSCC Encoder following the DeepJSCC architecture.

    Architecture:
        LayerNorm(D) → Linear(D, H) → GELU → Linear(H, k)

    The LayerNorm at the input stabilises the ViT feature distribution
    (which can vary widely in magnitude across layers/tokens).
    Two linear layers with a GELU non-linearity allow the encoder to exploit
    non-linear statistical structure that a single linear projection cannot.

    Power normalisation is applied *outside* this module.

    Args:
        feat_dim    : Input feature dimension D (e.g. 1024 for ViT-Large).
        channel_dim : Output channel symbol dimension k.
        hidden_dim  : Hidden layer width H.  Defaults to ``feat_dim``.
    """

    def __init__(self, feat_dim: int, channel_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.feat_dim    = feat_dim
        self.channel_dim = channel_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, channel_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat : (B, N, D)
        Returns:
            z    : (B, N, k)
        """
        return self.encoder(feat)

    @property
    def compression_ratio(self) -> float:
        """k / D — fraction of the original feature dimension retained."""
        return self.channel_dim / self.feat_dim

    def extra_repr(self) -> str:
        return (
            f'feat_dim={self.feat_dim}, hidden_dim={self.hidden_dim}, '
            f'channel_dim={self.channel_dim}, ratio={self.compression_ratio:.4f}'
        )


class JSCCDecoder(nn.Module):
    """
    Non-linear JSCC Decoder following the DeepJSCC architecture.

    Architecture:
        Linear(k, H) → GELU → Linear(H, D) → LayerNorm(D)

    The LayerNorm at the output re-normalises the reconstructed features into
    the same statistical space as the original ViT encoder output.

    Args:
        channel_dim : Input channel symbol dimension k.
        feat_dim    : Output feature dimension D (e.g. 1024 for ViT-Large).
        hidden_dim  : Hidden layer width H.  Defaults to ``feat_dim``.
    """

    def __init__(self, channel_dim: int, feat_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.channel_dim = channel_dim
        self.feat_dim    = feat_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim

        self.decoder = nn.Sequential(
            nn.Linear(channel_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat    : (B, N, k)  received channel symbols (unit-power scale)
        Returns:
            feat_hat : (B, N, D)  reconstructed features
        """
        return self.decoder(z_hat)

    def extra_repr(self) -> str:
        return (
            f'channel_dim={self.channel_dim}, hidden_dim={self.hidden_dim}, '
            f'feat_dim={self.feat_dim}'
        )
