# --------------------------------------------------------
# DUSt3R + Semantic Communication
#
# Wraps AsymmetricCroCo3DStereo with a SemComBlock inserted between
# the ViT encoder and the cross-view decoder.
# --------------------------------------------------------
import os
import torch

from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.semcom import SemComBlock
from dust3r.utils.misc import freeze_all_params


class DUSt3RSemCom(AsymmetricCroCo3DStereo):
    """
    DUSt3R model with a Semantic Communication channel injected between
    the ViT encoder and the cross-view decoder.

        Image1, Image2
             │
        [DUSt3R ViT Encoder]
             │
        feat1, feat2  (B, N, D)
             │
        [SemComBlock]   ← power-norm → channel (→ JSCC dec)
             │
        feat1_hat, feat2_hat
             │
        [DUSt3R Decoder + Head]
             │
        pred1, pred2

    Args:
        semcom : SemComBlock or None.
                 ``None`` → behaves identically to the original DUSt3R.
        **kwargs : forwarded to AsymmetricCroCo3DStereo.__init__.
    """

    def __init__(self, semcom=None, **kwargs):
        super().__init__(**kwargs)
        self.semcom = semcom

    def forward(self, view1, view2):
        # ── Encoder ──────────────────────────────────────────────────────────
        encoder_frozen = not next(self.patch_embed.parameters()).requires_grad
        with torch.set_grad_enabled(not encoder_frozen):
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = \
                self._encode_symmetrized(view1, view2)

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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _transplant(base, semcom) -> DUSt3RSemCom:
    """Transplant backbone weights + semcom block into a DUSt3RSemCom instance."""
    model = DUSt3RSemCom.__new__(DUSt3RSemCom)
    model.__dict__.update(base.__dict__)
    model.semcom = semcom
    model.__class__ = DUSt3RSemCom
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def load_semcom_model(
    dust3r_path: str,
    device: str,
    snr_db: float = float('inf'),
    channel: str = 'awgn',
    jscc_path: str | None = None,
    verbose: bool = True,
) -> DUSt3RSemCom:
    """
    Load a DUSt3R + SemCom model for inference.

    Behaviour depends on the supplied arguments:

    +--------------+----------+------------------------------------------------+
    | jscc_path    | snr_db   | Effect                                         |
    +==============+==========+================================================+
    | None         | inf      | Clean DUSt3R baseline (no SemCom block)        |
    | None         | finite   | Noise-only: identity JSCC, no learnable params |
    | JSCC ckpt    | any      | DeepJSCC with trained encoder/decoder;         |
    |              |          | backbone weights come from dust3r_path         |
    | Full ckpt    | any      | Jointly fine-tuned model (backbone + SemCom)   |
    +--------------+----------+------------------------------------------------+

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        device      : ``'cuda'`` or ``'cpu'``.
        snr_db      : Inference SNR in dB.  Overrides any value stored in a
                      loaded checkpoint.
        channel     : ``'awgn'`` or ``'rayleigh'`` — used only in noise-only mode.
        jscc_path   : Optional path to a JSCC-only or full-model checkpoint.
        verbose     : Print loading messages.

    Returns:
        model : DUSt3RSemCom in eval mode on *device*.
    """
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    if jscc_path is None:
        # ── No checkpoint: noise-only or clean baseline ────────────────────
        if snr_db == float('inf'):
            semcom = None
            if verbose:
                print('[SemCom] SNR=inf → clean DUSt3R baseline (no channel block)')
        else:
            semcom = SemComBlock(channel=channel, snr_db=snr_db)
            if verbose:
                print(f'[SemCom] Noise-only | channel={channel}, SNR={snr_db} dB')
        model = _transplant(base, semcom)

    else:
        # ── Load from checkpoint ──────────────────────────────────────────
        ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
        cfg  = ckpt['config']

        semcom = SemComBlock(
            channel=cfg['channel'],
            snr_db=snr_db,
            feat_dim=cfg['feat_dim'],
            channel_dim=cfg['channel_dim'],
            hidden_dim=cfg.get('hidden_dim'),
        )

        if verbose:
            if cfg['channel_dim'] is None:
                print(
                    f'[SemCom] Loaded from {jscc_path}\n'
                    f'  channel={cfg["channel"]}, identity (no JSCC compression), '
                    f'inference SNR={snr_db} dB'
                )
            else:
                ratio = cfg['channel_dim'] / cfg['feat_dim']
                print(
                    f'[SemCom] Loaded from {jscc_path}\n'
                    f'  channel={cfg["channel"]}, channel_dim={cfg["channel_dim"]}'
                    f' (ratio={ratio:.3f}), inference SNR={snr_db} dB'
                )

        model = _transplant(base, semcom)

        if 'model' in ckpt:
            # Full fine-tuned checkpoint (backbone + SemCom jointly trained)
            model.load_state_dict(ckpt['model'], strict=True)
            model.semcom.snr_db = snr_db
        else:
            # JSCC-only checkpoint (frozen backbone training)
            semcom.load_state_dict(ckpt['jscc_state_dict'])

    return model.to(device).eval()


def build_semcom_model(
    dust3r_path: str,
    device: str = 'cpu',
    freeze: str = 'none',
    snr_db: float = 10.0,
    channel: str = 'awgn',
    channel_dim: int = 512,
    hidden_dim: int | None = None,
    feat_dim: int = 1024,
    jscc_path: str | None = None,
    verbose: bool = True,
) -> tuple:
    """
    Build a DUSt3RSemCom model ready for training.

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        device      : ``'cuda'`` or ``'cpu'``.
        freeze      : Backbone freeze strategy:
                        ``'encoder'`` — freeze ViT patch_embed + enc_blocks;
                                        train JSCC + decoder + heads.
                        ``'none'``    — all parameters trainable (full joint).
        snr_db      : Default SNR in dB (typically overridden per-batch during training).
        channel     : ``'awgn'`` or ``'rayleigh'``.
        channel_dim : JSCC channel symbol dimension k.
        hidden_dim  : JSCC MLP hidden width H (defaults to feat_dim).
        feat_dim    : Encoder feature dimension D.
        jscc_path   : Optional checkpoint for JSCC warm-start.  Accepts both
                      JSCC-only (``jscc_state_dict`` key) and full-model
                      (``model`` key) checkpoints.
        verbose     : Print loading messages.

    Returns:
        (model, jscc_config)
        model       : DUSt3RSemCom in train mode on *device*.
        jscc_config : dict with JSCC hyper-parameters (for checkpoint saving).
    """
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    if jscc_path is not None and os.path.exists(jscc_path):
        ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
        if 'jscc_state_dict' not in ckpt:
            raise ValueError(
                f'Cannot read JSCC weights from {jscc_path!r}: '
                '"jscc_state_dict" key not found.'
            )
        cfg = ckpt['config']
        semcom = SemComBlock(
            channel=cfg['channel'],
            snr_db=snr_db,
            feat_dim=cfg['feat_dim'],
            channel_dim=cfg['channel_dim'],
            hidden_dim=cfg.get('hidden_dim'),
        )
        semcom.load_state_dict(ckpt['jscc_state_dict'])
        jscc_config = {
            'feat_dim':    cfg['feat_dim'],
            'channel_dim': cfg['channel_dim'],
            'hidden_dim':  cfg.get('hidden_dim'),
            'channel':     cfg['channel'],
        }
        if verbose:
            print(f'[SemCom] JSCC warm-start from {jscc_path}')
    else:
        semcom = SemComBlock(
            channel=channel,
            snr_db=snr_db,
            feat_dim=feat_dim,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim,
        )
        jscc_config = {
            'feat_dim':    feat_dim,
            'channel_dim': channel_dim,
            'hidden_dim':  hidden_dim,
            'channel':     channel,
        }
        if verbose:
            if channel_dim is None:
                print(f'[SemCom] Identity channel (no JSCC compression), channel={channel}')
            else:
                print(f'[SemCom] Fresh JSCC: k={channel_dim}, channel={channel}')

    model = _transplant(base, semcom)

    if freeze == 'encoder':
        freeze_all_params([model.mask_token, model.patch_embed, model.enc_blocks])
        if verbose:
            print('[SemCom] Frozen: patch_embed + enc_blocks (ViT encoder)')
    elif freeze == 'none':
        if verbose:
            print('[SemCom] No freezing — full joint training')
    else:
        raise ValueError(f'Unknown freeze={freeze!r}. Choose "none" or "encoder".')

    if verbose:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'[SemCom] Parameters: {total / 1e6:.1f}M total, '
            f'{trainable / 1e6:.1f}M trainable'
        )

    return model.to(device).train(), jscc_config
