# --------------------------------------------------------
# DUSt3R + Semantic Communication (Phase A)
#
# Inherits AsymmetricCroCo3DStereo and injects a SemCom block
# between the encoder and the decoder.  No retraining is needed
# for Phase A (identity JSCC, noise-only channel).
# --------------------------------------------------------
import torch

from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.semcom import PhaseASemCom


class AsymmetricCroCo3DStereo_SemCom(AsymmetricCroCo3DStereo):
    """
    DUSt3R model augmented with a Semantic Communication channel.

    The SemCom block is inserted between the shared encoder and the
    cross-view decoder:

        Image1, Image2
             |
        [DUSt3R Encoder]          (frozen pretrained weights)
             |
        feat1, feat2  (B, N, D)
             |
        [SemCom Block]            <-- inserted here
          power-norm -> channel -> inverse-scale
             |
        feat1_hat, feat2_hat
             |
        [DUSt3R Decoder + Head]   (frozen pretrained weights)
             |
        pred1, pred2

    Args:
        semcom_block : nn.Module or None
            If None, behaves exactly like the original DUSt3R (no channel).
            Pass a PhaseASemCom instance (or any nn.Module with the same
            signature) to enable the SemCom channel.
        **kwargs     : forwarded to AsymmetricCroCo3DStereo.__init__
    """

    def __init__(self, semcom_block=None, **kwargs):
        super().__init__(**kwargs)
        self.semcom = semcom_block

    def forward(self, view1, view2):
        # ── Encoder ──────────────────────────────────────────────────────────
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # ── SemCom Channel ───────────────────────────────────────────────────
        if self.semcom is not None:
            feat1 = self.semcom(feat1)
            feat2 = self.semcom(feat2)

        # ── Decoder + Head ───────────────────────────────────────────────────
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast('cuda', enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2


# ── Convenience loader ────────────────────────────────────────────────────────

def load_semcom_model(model_path: str, device: str,
                      snr_db: float = float('inf'),
                      channel: str = 'awgn',
                      verbose: bool = True):
    """
    Load a pretrained DUSt3R checkpoint and wrap it with a SemCom channel.

    Args:
        model_path : path to the .pth checkpoint
        device     : 'cuda' or 'cpu'
        snr_db     : SNR in dB for the AWGN/Rayleigh channel.
                     float('inf') → no noise (clean baseline identical to
                     original DUSt3R).
        channel    : 'awgn' or 'rayleigh'
        verbose    : print loading messages

    Returns:
        model : AsymmetricCroCo3DStereo_SemCom on the requested device,
                in eval mode with grad disabled.
    """
    # 1. Load the base DUSt3R model (uses the existing load_model logic)
    base = load_model(model_path, device='cpu', verbose=verbose)

    # 2. Build the SemCom block
    if snr_db == float('inf'):
        semcom_block = None
        if verbose:
            print(f'[SemCom] SNR = inf  →  noiseless baseline (no channel block)')
    else:
        semcom_block = PhaseASemCom(snr_db=snr_db, channel=channel)
        if verbose:
            print(f'[SemCom] SNR = {snr_db} dB, channel = {channel}')

    # 3. Transplant weights into the SemCom-aware subclass
    #    (state dict is compatible because we only added self.semcom)
    model = AsymmetricCroCo3DStereo_SemCom.__new__(AsymmetricCroCo3DStereo_SemCom)
    model.__dict__.update(base.__dict__)          # copy all attributes
    model.semcom = semcom_block                   # attach SemCom block
    # fix the class so isinstance checks work correctly
    model.__class__ = AsymmetricCroCo3DStereo_SemCom

    model = model.to(device).eval()
    return model
