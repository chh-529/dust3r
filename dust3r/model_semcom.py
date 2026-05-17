# --------------------------------------------------------
# DUSt3R + Semantic Communication (Phase A)
#
# Inherits AsymmetricCroCo3DStereo and injects a SemCom block
# between the encoder and the decoder.  No retraining is needed
# for Phase A (identity JSCC, noise-only channel).
# --------------------------------------------------------
import torch

from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.semcom import PhaseASemCom, PhaseBSemCom


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


def load_semcom_model_phaseB(
    dust3r_path: str,
    jscc_path: str,
    device: str,
    snr_db: float = 10.0,
    verbose: bool = True,
):
    """
    Load a pretrained DUSt3R checkpoint together with a Phase B JSCC checkpoint
    (saved by ``train_semcom_phaseB.py``).

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        jscc_path   : Path to the JSCC checkpoint produced by training.
        device      : ``'cuda'`` or ``'cpu'``.
        snr_db      : Inference-time SNR in dB.  May differ from the SNR used
                      during training (useful for evaluating generalisation).
        verbose     : Print loading messages.

    Returns:
        model : AsymmetricCroCo3DStereo_SemCom with Phase B JSCC, in eval mode.

    Checkpoint format (written by train_semcom_phaseB.py):
        {
            'jscc_state_dict' : PhaseBSemCom state dict,
            'config'          : {'feat_dim', 'channel_dim', 'channel',
                                 'snr_db' (training SNR or range)},
            'epoch'           : last completed epoch (int),
            'train_losses'    : list of per-epoch mean losses,
        }
    """
    import torch

    # 1. Load DUSt3R backbone (frozen weights)
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    # 2. Load JSCC checkpoint
    jscc_ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
    cfg = jscc_ckpt['config']

    if verbose:
        ratio = cfg['channel_dim'] / cfg['feat_dim']
        trained_snr = cfg.get('snr_db', cfg.get('snr_range', 'N/A'))
        hidden_dim = cfg.get('hidden_dim', cfg['feat_dim'])
        print(
            f'[SemCom Phase B] Loaded JSCC from {jscc_path}\n'
            f'  feat_dim={cfg["feat_dim"]}, hidden_dim={hidden_dim}, '
            f'channel_dim={cfg["channel_dim"]} '
            f'(ratio={ratio:.3f}), channel={cfg["channel"]}\n'
            f'  Trained at SNR={trained_snr} dB  |  Inference SNR={snr_db} dB'
        )

    # 3. Rebuild PhaseBSemCom and load weights
    semcom = PhaseBSemCom(
        feat_dim=cfg['feat_dim'],
        channel_dim=cfg['channel_dim'],
        hidden_dim=cfg.get('hidden_dim', None),   # None → defaults to feat_dim
        snr_db=snr_db,          # inference SNR (may differ from training SNR)
        channel=cfg['channel'],
    )
    semcom.load_state_dict(jscc_ckpt['jscc_state_dict'])

    # 4. Transplant into the SemCom-aware model class
    model = AsymmetricCroCo3DStereo_SemCom.__new__(AsymmetricCroCo3DStereo_SemCom)
    model.__dict__.update(base.__dict__)
    model.semcom = semcom
    model.__class__ = AsymmetricCroCo3DStereo_SemCom

    return model.to(device).eval()
