import math
import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel
from .jscc import JSCCEncoder, JSCCDecoder


def _power_normalize(z: torch.Tensor):
    """
    Normalize z so that the average power per symbol equals 1.

    z   : (B, N, D)
    Returns (z_normalized, rms) where rms has shape (B, 1, 1).
    rms is the per-sample RMS before normalization; kept for inverse scaling.
    """
    B = z.shape[0]
    flat = z.view(B, -1)                          # (B, N*D)
    k = flat.shape[1]
    rms = flat.norm(dim=1).unsqueeze(-1).unsqueeze(-1) / math.sqrt(k)  # (B,1,1)
    z_norm = z / (rms + 1e-8)
    return z_norm, rms


class SemComBlock(nn.Module):
    """
    Semantic Communication block for DUSt3R.

    Sits between the ViT encoder and cross-view decoder.  Two operating modes
    are selected by ``channel_dim``:

    Identity mode  (``channel_dim=None``)
        No compression / decompression.  The feature tensor is power-normalised,
        passed through the physical channel, then inverse-scaled back.
        No learnable parameters.  Useful as a noise-injection baseline.

        Pipeline:
            feat  (B, N, D)
              └─ power_norm → channel → inverse_scale → feat_hat  (B, N, D)

    DeepJSCC mode  (``channel_dim=k``)
        Learnable non-linear encoder / decoder (MLP + GELU) compress the
        features to k symbols, transmit over the channel, then reconstruct.

        Pipeline:
            feat  (B, N, D)
              └─ JSCCEncoder(D→k) → power_norm → channel
                 → JSCCDecoder(k→D) → feat_hat  (B, N, D)

    Args:
        channel     : Physical channel model — ``'awgn'`` or ``'rayleigh'``.
        snr_db      : Default SNR in dB.  Use ``float('inf')`` for noiseless.
                      Can be overridden per ``forward`` call.
        feat_dim    : Input feature dimension D (e.g. 1024 for ViT-Large).
        channel_dim : Channel symbol dimension k.  ``None`` → identity mode
                      (no compression, no learnable parameters).
        hidden_dim  : MLP hidden width H for the JSCC encoder/decoder.
                      Defaults to ``feat_dim``.
    """

    def __init__(
        self,
        channel: str = 'awgn',
        snr_db: float = 10.0,
        feat_dim: int = 1024,
        channel_dim: int | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.snr_db     = snr_db
        self.feat_dim   = feat_dim
        self.channel_dim = channel_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim

        # Physical channel
        channel_lower = channel.lower()
        if channel_lower == 'awgn':
            self.channel = AWGNChannel()
        elif channel_lower == 'rayleigh':
            self.channel = RayleighChannel()
        else:
            raise ValueError(
                f'Unknown channel {channel!r}. Choose "awgn" or "rayleigh".'
            )

        # Learnable JSCC encoder / decoder (DeepJSCC mode only)
        if channel_dim is not None:
            self.jscc_encoder = JSCCEncoder(feat_dim, channel_dim, self.hidden_dim)
            self.jscc_decoder = JSCCDecoder(channel_dim, feat_dim, self.hidden_dim)
        else:
            self.jscc_encoder = None
            self.jscc_decoder = None

    def forward(
        self,
        feat: torch.Tensor,
        snr_db: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            feat   : (B, N, D)  encoder output tokens.
            snr_db : SNR override in dB.  Uses ``self.snr_db`` when ``None``.
        Returns:
            feat_hat : (B, N, D)  reconstructed tokens after channel.
        """
        if snr_db is None:
            snr_db = self.snr_db

        if self.jscc_encoder is not None:
            # ── DeepJSCC mode ─────────────────────────────────────────────
            z = self.jscc_encoder(feat)
            z_norm, _ = _power_normalize(z)
            z_hat = self.channel(z_norm, snr_db)
            return self.jscc_decoder(z_hat)
        else:
            # ── Identity mode ─────────────────────────────────────────────
            z, rms = _power_normalize(feat)
            z_hat = self.channel(z, snr_db)
            return z_hat * (rms + 1e-8)

    @property
    def compression_ratio(self) -> float:
        """k / D — fraction of the original dimension retained (1.0 in identity mode)."""
        if self.channel_dim is None:
            return 1.0
        return self.channel_dim / self.feat_dim

    def extra_repr(self) -> str:
        mode = (
            f'feat_dim={self.feat_dim}, channel_dim={self.channel_dim}, '
            f'ratio={self.compression_ratio:.4f}'
            if self.channel_dim is not None
            else 'identity (no compression)'
        )
        return (
            f'{mode}, snr_db={self.snr_db}, '
            f'channel={self.channel.__class__.__name__}'
        )
