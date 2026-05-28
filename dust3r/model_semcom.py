# --------------------------------------------------------
# DUSt3R + Semantic Communication (Phase A / B / C)
#
# Inherits AsymmetricCroCo3DStereo and injects a SemCom block
# between the encoder and the decoder.  No retraining is needed
# for Phase A (identity JSCC, noise-only channel).
# Phase B trains the JSCC encoder/decoder only (frozen backbone).
# Phase C jointly trains the full pipeline end-to-end.
# --------------------------------------------------------
import os
import torch

from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.semcom import PhaseASemCom, PhaseBSemCom
from dust3r.utils.misc import freeze_all_params


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
        # If encoder is frozen, skip storing activations to save GPU memory
        encoder_frozen = not next(self.patch_embed.parameters()).requires_grad
        with torch.set_grad_enabled(not encoder_frozen):
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


# ── Phase C helpers ──────────────────────────────────────────────────────────

def _transplant(base, semcom) -> 'AsymmetricCroCo3DStereo_SemCom':
    """Transplant *base* backbone weights + *semcom* into a SemCom model."""
    model = AsymmetricCroCo3DStereo_SemCom.__new__(AsymmetricCroCo3DStereo_SemCom)
    model.__dict__.update(base.__dict__)
    model.semcom = semcom
    model.__class__ = AsymmetricCroCo3DStereo_SemCom
    return model


def build_semcom_model_phaseC(
    dust3r_path: str,
    jscc_path: str | None = None,
    device: str = 'cpu',
    freeze: str = 'encoder',
    snr_db: float = 10.0,
    channel: str = 'awgn',
    channel_dim: int = 512,
    hidden_dim: int | None = None,
    feat_dim: int = 1024,
    verbose: bool = True,
):
    """
    Build a Phase C model ready for end-to-end joint training.

    Phase C trains the full pipeline:
        ViT encoder → JSCC encoder → channel → JSCC decoder → ViT decoder → head

    The JSCC can optionally be warm-started from a Phase B checkpoint
    (``jscc_path``).  The backbone freeze strategy is controlled by
    ``freeze``.

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        jscc_path   : (optional) Phase B or Phase C checkpoint for JSCC
                      warm-start.  If None, a fresh JSCC is created.
        device      : 'cuda' or 'cpu'.
        freeze      : Backbone freeze strategy:
                        'none'    – all parameters trainable (full joint)
                        'encoder' – freeze ViT patch_embed + enc_blocks;
                                    only decoder + heads + JSCC are trained.
        snr_db      : Default SNR in dB (will be overridden per batch during training).
        channel     : 'awgn' or 'rayleigh'.
        channel_dim : JSCC channel symbol dimension k (ignored when jscc_path given).
        hidden_dim  : JSCC MLP hidden width (ignored when jscc_path given).
        feat_dim    : Encoder feature dimension D (ignored when jscc_path given).
        verbose     : Print loading messages.

    Returns:
        (model, jscc_config)
        model       : AsymmetricCroCo3DStereo_SemCom in train mode on *device*.
        jscc_config : dict with JSCC hyper-parameters (for checkpoint saving).
    """
    # 1. Load backbone
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    # 2. Build JSCC — warm-start from checkpoint or fresh
    if jscc_path is not None and os.path.exists(jscc_path):
        ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
        if 'jscc_state_dict' not in ckpt:
            raise ValueError(
                f'Cannot read JSCC weights from {jscc_path!r}: '
                '"jscc_state_dict" key not found.'
            )
        cfg = ckpt['config']
        semcom = PhaseBSemCom(
            feat_dim=cfg['feat_dim'],
            channel_dim=cfg['channel_dim'],
            hidden_dim=cfg.get('hidden_dim', None),
            snr_db=snr_db,
            channel=cfg['channel'],
        )
        semcom.load_state_dict(ckpt['jscc_state_dict'])
        jscc_config = {
            'feat_dim':    cfg['feat_dim'],
            'channel_dim': cfg['channel_dim'],
            'hidden_dim':  cfg.get('hidden_dim', None),
            'channel':     cfg['channel'],
        }
        if verbose:
            print(f'[SemCom Phase C] JSCC warm-start from {jscc_path}')
    else:
        semcom = PhaseBSemCom(
            feat_dim=feat_dim,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim,
            snr_db=snr_db,
            channel=channel,
        )
        jscc_config = {
            'feat_dim':    feat_dim,
            'channel_dim': channel_dim,
            'hidden_dim':  hidden_dim,
            'channel':     channel,
        }
        if verbose:
            print(f'[SemCom Phase C] Fresh JSCC: k={channel_dim}, channel={channel}')

    # 3. Transplant
    model = _transplant(base, semcom)

    # 4. Apply freeze strategy
    if freeze == 'encoder':
        freeze_all_params([model.mask_token, model.patch_embed, model.enc_blocks])
        if verbose:
            print('[SemCom Phase C] Frozen: patch_embed + enc_blocks (ViT encoder)')
    elif freeze == 'none':
        if verbose:
            print('[SemCom Phase C] No freezing — full joint training')
    else:
        raise ValueError(f'Unknown freeze={freeze!r}. Choose "none" or "encoder".')

    if verbose:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'[SemCom Phase C] Parameters: {total / 1e6:.1f}M total, '
            f'{trainable / 1e6:.1f}M trainable'
        )

    return model.to(device).train(), jscc_config


def load_semcom_model_phaseC(
    dust3r_path: str,
    phasec_path: str,
    device: str,
    snr_db: float = 10.0,
    verbose: bool = True,
):
    """
    Load a saved Phase C checkpoint for inference.

    Phase C checkpoints contain the full model state dict (backbone + JSCC),
    saved by ``train_semcom_phaseC.py``.

    Args:
        dust3r_path : Original DUSt3R .pth (for model architecture).
        phasec_path : Phase C checkpoint produced by train_semcom_phaseC.py.
        device      : 'cuda' or 'cpu'.
        snr_db      : Inference-time SNR in dB.
        verbose     : Print loading messages.

    Returns:
        model : AsymmetricCroCo3DStereo_SemCom in eval mode on *device*.

    Checkpoint format (written by train_semcom_phaseC.py):
        {
            'model'          : full state dict (backbone + JSCC),
            'jscc_state_dict': JSCC-only state dict (for backward compat),
            'config'         : {'feat_dim', 'channel_dim', 'hidden_dim',
                                'channel', 'freeze', 'snr_db', ...},
            'epoch'          : last completed epoch,
            'train_losses'   : list of per-epoch losses,
        }
    """
    # 1. Architecture from original DUSt3R backbone
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    # 2. Load Phase C checkpoint
    ckpt = torch.load(phasec_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']

    if verbose:
        trained_snr = cfg.get('snr_db', 'N/A')
        print(
            f'[SemCom Phase C] Loaded from {phasec_path}\n'
            f'  feat_dim={cfg["feat_dim"]}, channel_dim={cfg["channel_dim"]}, '
            f'channel={cfg["channel"]}, epoch={ckpt.get("epoch", "?")}\n'
            f'  Trained SNR={trained_snr}  |  Inference SNR={snr_db} dB'
        )

    # 3. Rebuild PhaseBSemCom shell (weights loaded from ckpt['model'])
    semcom = PhaseBSemCom(
        feat_dim=cfg['feat_dim'],
        channel_dim=cfg['channel_dim'],
        hidden_dim=cfg.get('hidden_dim', None),
        snr_db=snr_db,
        channel=cfg['channel'],
    )

    # 4. Transplant shell
    model = _transplant(base, semcom)

    # 5. Load full state dict (backbone + JSCC)
    model.load_state_dict(ckpt['model'], strict=True)

    # 6. Override inference SNR (state dict may contain training SNR)
    model.semcom.snr_db = snr_db

    return model.to(device).eval()
