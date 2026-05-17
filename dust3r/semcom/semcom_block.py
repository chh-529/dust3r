import math
import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel
from .jscc import LinearJSCCEncoder, LinearJSCCDecoder, JSCCEncoder, JSCCDecoder


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
    # RMS per symbol:  sqrt( ||z||^2 / k )
    rms = flat.norm(dim=1).unsqueeze(-1).unsqueeze(-1) / math.sqrt(k)  # (B,1,1)
    z_norm = z / (rms + 1e-8)
    return z_norm, rms


class PhaseASemCom(nn.Module):
    """
    Phase A Semantic Communication block — no JSCC learning.

    The JSCC Encoder and Decoder are both identity mappings.
    Only power normalization + physical channel noise are applied.

    Pipeline:
        feat  (B, N, D)
          |-- power normalize --> z  (B, N, D),  avg power/symbol = 1
          |-- channel noise   --> z_hat
          |-- inverse scale   --> feat_hat  (B, N, D)

    Args:
        snr_db    : float  SNR in dB (use float('inf') for noise-free baseline)
        channel   : str    'awgn' or 'rayleigh'
    """

    def __init__(self, snr_db: float = 10.0, channel: str = 'awgn'):
        super().__init__()
        self.snr_db = snr_db
        channel = channel.lower()
        if channel == 'awgn':
            self.channel = AWGNChannel()
        elif channel == 'rayleigh':
            self.channel = RayleighChannel()
        else:
            raise ValueError(f'Unknown channel type "{channel}". Choose "awgn" or "rayleigh".')

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat : (B, N, D)  encoder output tokens
        Returns:
            feat_hat : (B, N, D)  reconstructed tokens after channel
        """
        # 1. Power normalization
        z, rms = _power_normalize(feat)

        # 2. Physical channel (adds noise or passes through if SNR=inf)
        z_hat = self.channel(z, self.snr_db)

        # 3. Inverse scale: map received symbols back to original feature magnitude
        feat_hat = z_hat * (rms + 1e-8)
        return feat_hat

    def extra_repr(self) -> str:
        return f'snr_db={self.snr_db}, channel={self.channel.__class__.__name__}'


# ── Phase B ───────────────────────────────────────────────────────────────────

class PhaseBSemCom(nn.Module):
    """
    Phase B Semantic Communication block — non-linear DeepJSCC with learnable
    encoder and decoder.

    Pipeline:
        feat  (B, N, D)
          |-- JSCC Encoder  --> z       (B, N, k)   non-linear D→k (LayerNorm+MLP+GELU)
          |-- power norm    --> z_norm              unit avg power / symbol
          |-- channel       --> z_hat               AWGN or Rayleigh fading
          |-- JSCC Decoder  --> feat_hat (B, N, D)  non-linear k→D (MLP+GELU+LayerNorm)

    Only ``jscc_encoder`` and ``jscc_decoder`` are learnable; the channel is
    a fixed stochastic function.  During training keep DUSt3R weights frozen
    and optimise only this module.

    Architecture follows DeepJSCC (Bourtsoulatze et al. 2019) adapted for
    semantic features (Xie et al. DEEPSC 2021).

    Args:
        feat_dim    : Feature dimension D (default 1024 for ViT-Large).
        channel_dim : Channel symbol dimension k (k ≤ D for compression).
        hidden_dim  : MLP hidden width H (default: same as feat_dim).
        snr_db      : Default SNR in dB.  Can be overridden per forward call.
        channel     : ``'awgn'`` or ``'rayleigh'``.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        channel_dim: int = 512,
        hidden_dim: int | None = None,
        snr_db: float = 10.0,
        channel: str = 'awgn',
    ):
        super().__init__()
        self.feat_dim   = feat_dim
        self.channel_dim = channel_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim
        self.snr_db = snr_db

        self.jscc_encoder = JSCCEncoder(feat_dim, channel_dim, self.hidden_dim)
        self.jscc_decoder = JSCCDecoder(channel_dim, feat_dim, self.hidden_dim)

        channel_lower = channel.lower()
        if channel_lower == 'awgn':
            self.channel = AWGNChannel()
        elif channel_lower == 'rayleigh':
            self.channel = RayleighChannel()
        else:
            raise ValueError(
                f'Unknown channel: {channel!r}. Choose "awgn" or "rayleigh".'
            )

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

        # 1. JSCC encode: D → k
        z = self.jscc_encoder(feat)

        # 2. Power normalise (unit average power per symbol)
        z_norm, _ = _power_normalize(z)

        # 3. Physical channel (AWGN or Rayleigh)
        z_hat = self.channel(z_norm, snr_db)

        # 4. JSCC decode: k → D
        feat_hat = self.jscc_decoder(z_hat)
        return feat_hat

    @property
    def compression_ratio(self) -> float:
        return self.jscc_encoder.compression_ratio

    def extra_repr(self) -> str:
        return (
            f'feat_dim={self.feat_dim}, channel_dim={self.channel_dim}, '
            f'ratio={self.compression_ratio:.4f}, '
            f'snr_db={self.snr_db}, channel={self.channel.__class__.__name__}'
        )
