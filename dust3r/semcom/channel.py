import math
import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Assumes the input z has been power-normalized to unit average power
    per symbol (i.e. E[|z_i|^2] = 1).

    Noise power is set so that SNR = signal_power / noise_power
                                   = 1 / sigma^2
    => sigma = 1 / sqrt(SNR_linear)
    """

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Args:
            z      : (B, N, k) power-normalized channel symbols
            snr_db : SNR in dB; use float('inf') for a noiseless channel
        Returns:
            z_hat  : (B, N, k) received symbols
        """
        if snr_db == float('inf'):
            return z
        snr_linear = 10 ** (snr_db / 10.0)
        sigma = math.sqrt(1.0 / snr_linear)
        noise = torch.randn_like(z) * sigma
        return z + noise


class RayleighChannel(nn.Module):
    """
    Flat Rayleigh fading channel with perfect CSI at receiver (CSIR).

    h ~ CN(0, 1)  =>  received = h * z + n
    With CSIR, the receiver divides by h to equalize:
      z_hat = z + n / h  (approximate; we use the noise-scaling form)

    For simplicity we use the equivalent real-valued model:
      z_hat = |h| * z + n,   then equalize by dividing by |h|
    This gives effective noise  n_eff ~ N(0, sigma^2 / |h|^2).
    """

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Args:
            z      : (B, N, k) power-normalized channel symbols
            snr_db : SNR in dB; use float('inf') for a noiseless channel
        Returns:
            z_hat  : (B, N, k) equalized received symbols
        """
        if snr_db == float('inf'):
            return z
        snr_linear = 10 ** (snr_db / 10.0)
        sigma = math.sqrt(1.0 / snr_linear)

        # Rayleigh fading envelope: |h| ~ Rayleigh(1/sqrt(2))
        # h = h_re + j*h_im, h_re,h_im ~ N(0, 0.5)
        h = torch.sqrt(
            torch.randn_like(z) ** 2 / 2 + torch.randn_like(z) ** 2 / 2
        )  # (B, N, k), |h| ~ Rayleigh

        noise = torch.randn_like(z) * sigma

        # CSIR equalization: divide by h
        z_hat = z + noise / (h + 1e-8)
        return z_hat
