import math
import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel


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
