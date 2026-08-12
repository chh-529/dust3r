"""
The end-to-end system: DUSt3R + semantic codec + channel.

    image -> ViT encoder -> [codec encoder] -> z -> CHANNEL -> y -> [codec decoder]
          -> feat_hat -> DUSt3R cross-view decoder -> heads -> pointmaps
                      -> (optional) RGB head       -> colours

SemComDust3R is a drop-in replacement for AsymmetricCroCo3DStereo: it has the same
forward(view1, view2) -> (res1, res2) signature, so dust3r.inference.loss_of_one_batch,
dust3r.inference.inference and train.py all work unchanged.

Two configurations, chosen by how many codecs you pass:

    n_user == 1   one transmitter sends BOTH views as a single V=2 signal.
                  Use a SingleChannel. This is the Phase 1/2 setup: it exercises the
                  whole pipeline with no multiple-access ambiguity.

    n_user == 2   transmitter 1 sends view1, transmitter 2 sends view2, superimposed on
                  the same symbols. Use a MultiUplinkChannel. The receiver gets one
                  stream y and each decoder pulls its own view out of it. This is the
                  Phase 3 setup and the architecture in the diagram.

The receiver never sees pixels
------------------------------
Everything the receiver is allowed to know is bundled into ReceiverSideInfo, which
holds image GEOMETRY only (sizes and orientations) and no pixel content. The patch
positions `pos` that DUSt3R's decoder needs are REBUILT at the receiver from that
geometry rather than carried over from the encoder, so there is no hidden noise-free
path from transmitter to receiver. This is enforced by construction: _receive() and
_decode_geometry() are given a ReceiverSideInfo and never a view dict.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .channel import Channel, MultiUplinkChannel, SingleChannel
from .codec import FeatureChannelDecoder, FeatureChannelEncoder
from .rgb_head import RGBHead


# Diagnostics from the most recent forward, per process. Written by
# SemComDust3R._record_stats(), read by the training criterion for logging. Never used
# in the model's own computation.
LATEST_STATS: Dict[str, float] = {}


@dataclass
class ReceiverSideInfo:
    """
    What the receiver legitimately knows about the images without receiving them.

    Sizes and orientations are properties of the capture setup, in the same category as
    the patch positions or the carrier frequency -- not content. There are NO pixels
    here, and that is the whole point of this type existing.
    """
    true_shape1: torch.Tensor           # (B, 2) per-sample (height, width) of view 1
    true_shape2: torch.Tensor           # (B, 2) same for view 2
    img_hw: Tuple[int, int]             # (H, W) of the padded image tensor, in pixels
    n_tokens: int                       # S, patch tokens per image


class SemComDust3R(nn.Module):
    """DUSt3R with a semantic codec and a channel spliced in after the ViT encoder."""

    def __init__(self,
                 dust3r: nn.Module,
                 encoders: Sequence[FeatureChannelEncoder],
                 decoders: Sequence[FeatureChannelDecoder],
                 channel: Channel,
                 power_constraints: Optional[Sequence[float]] = None,
                 freeze: str = 'dust3r',
                 rgb_head: Optional[RGBHead] = None,
                 verify_pos: bool = True):
        """
        Args:
            dust3r: an AsymmetricCroCo3DStereo. Its ViT encoder acts as the semantic
                encoder and its decoder + heads as the semantic decoder.
            encoders: one FeatureChannelEncoder per transmitter. Pass the SAME object
                twice to share weights; see the note in codec.make_codec_pair about
                what sharing costs you once two users superimpose.
            decoders: one FeatureChannelDecoder per user, same length as `encoders`.
            channel: a SingleChannel when n_user == 1, a MultiUplinkChannel when > 1.
            power_constraints: per-user average power per complex symbol. Defaults to
                1.0 for everyone. This is the power-allocation knob: unequal values are
                how you set up a near-far / NOMA experiment.
            freeze: 'dust3r' (default) trains only the codec; 'encoder' freezes just the
                ViT encoder; 'none' trains everything.
            rgb_head: optional appearance decoder, run on the RECEIVED features.
            verify_pos: on the first forward, check that the patch positions rebuilt at
                the receiver equal the ones the encoder actually used, and raise if not.
                Leave this on: a wrong `pos` produces no error, just silently wrong RoPE
                rotations and quietly worse results.
        """
        super().__init__()

        if len(encoders) != len(decoders):
            raise ValueError(f'{len(encoders) = } != {len(decoders) = }')
        n_user = len(encoders)
        if n_user not in (1, 2):
            raise ValueError(f'{n_user = } not supported (expected 1 or 2)')
        if n_user == 1 and isinstance(channel, MultiUplinkChannel):
            raise ValueError('n_user == 1 needs a SingleChannel-style channel')
        if n_user == 2 and isinstance(channel, SingleChannel):
            raise ValueError('n_user == 2 needs a MultiUplinkChannel')

        self.dust3r = dust3r
        self.n_user = n_user
        # ModuleList de-duplicates nothing, so sharing works: the same module listed
        # twice is registered once and has one set of parameters
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.channel = channel
        self.rgb_head = rgb_head

        if power_constraints is None:
            power_constraints = [1.0] * n_user
        if len(power_constraints) != n_user:
            raise ValueError(f'{len(power_constraints) = } != {n_user = }')
        self.power_constraints = list(power_constraints)
        for enc, p in zip(self.encoders, self.power_constraints):
            enc.power_constraint = p

        # Which patch embedding is in use decides how positions are laid out, and the
        # two DUSt3R variants disagree. Checked by class name rather than isinstance so
        # that importing semcom does not drag in the whole dust3r/croco stack.
        #   PatchEmbedDust3R : pos always follows the tensor grid, true_shape is ignored
        #   ManyAR_PatchEmbed: portrait images are laid out on a TRANSPOSED grid
        # The released DUSt3R_ViTLarge_BaseDecoder_512_dpt checkpoint uses the former.
        self.patch_embed_kind = type(dust3r.patch_embed).__name__
        self.transposes_portrait = (self.patch_embed_kind == 'ManyAR_PatchEmbed')

        self.set_freeze(freeze)
        self._verify_pos = verify_pos
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ setup

    def state_dict(self, *args, **kwargs):
        """
        Drop the frozen DUSt3R weights from the checkpoint.

        They are 571M parameters (~2.3 GB) that never change, and a bandwidth-ratio
        sweep writes one checkpoint per epoch per configuration, so keeping them would
        cost hundreds of gigabytes to store exactly what the pretrained file already
        holds. What is saved is only the codec (and the RGB head): ~12M parameters.

        Reloading therefore means rebuilding the model from the pretrained DUSt3R first
        and then loading these weights on top; croco's misc.load_model already passes
        strict=False, and semcom.eval does the same.
        """
        sd = super().state_dict(*args, **kwargs)
        if self.freeze == 'dust3r':
            return {k: v for k, v in sd.items() if not k.startswith('dust3r.')}
        return sd

    def set_freeze(self, freeze: str):
        self.freeze = freeze
        if freeze == 'none':
            targets = []
        elif freeze == 'encoder':
            targets = [self.dust3r.patch_embed, self.dust3r.enc_blocks, self.dust3r.mask_token]
        elif freeze == 'dust3r':
            targets = [self.dust3r]
        else:
            raise ValueError(f"invalid {freeze = }, expected 'none'/'encoder'/'dust3r'")

        for module in targets:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad = False

    # ------------------------------------------------- transmitter (sees images)

    def _encode_images(self, view1: dict, view2: dict):
        """
        ViT encoding. This is the only place images are read; it runs on the TX.

        When the encoder is frozen there is nothing upstream of it that needs gradients
        (its input is an image), so the whole 24-block ViT-L can run under no_grad. That
        removes its activations from memory entirely -- a large saving, and the reason
        the batch size can be reasonable at 512x384.

        The DUSt3R decoder and heads are a different matter: they are frozen too, but
        gradients must flow THROUGH them to reach the codec, so their activations have
        to be kept. That is the real memory bottleneck.
        """
        if self.freeze in ('dust3r', 'encoder'):
            with torch.no_grad():
                out = self.dust3r._encode_symmetrized(view1, view2)
            # detach so the codec sees a leaf: the features are constants w.r.t. training
            (s1, s2), (f1, f2), (p1, p2) = out
            return (s1, s2), (f1.detach(), f2.detach()), (p1, p2)
        return self.dust3r._encode_symmetrized(view1, view2)

    def _transmit(self, feat1: torch.Tensor, feat2: torch.Tensor) -> List[torch.Tensor]:
        """
        Features -> per-user power-constrained complex signals.

        Args:
            feat1, feat2: (B, S, D) each
        Returns:
            list of n_user complex tensors, each (B, L)
        """
        if self.n_user == 1:
            # one transmitter, both views in a single V=2 signal
            feat = torch.stack([feat1, feat2], dim=1)          # (B, 2, S, D)
            return [self.encoders[0](feat)]

        # one transmitter per view
        return [self.encoders[0](feat1.unsqueeze(1)),          # (B, 1, S, D)
                self.encoders[1](feat2.unsqueeze(1))]

    # ---------------------------------------------------------------- channel

    def _through_channel(self, signals: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns the single received stream y, (B, L). With n_user == 2 the users are
        superimposed, so every decoder gets the same y -- separation is learned, not
        performed here.
        """
        if hasattr(self.channel, 'resample_snr') and self.training:
            self.channel.resample_snr()

        if self.n_user == 1:
            return self.channel.interfere(signals[0])

        stacked = torch.stack(signals, dim=0)                  # (n_tx, B, L)
        return self.channel.interfere(stacked, user_dim_index=0)

    # ------------------------------------------------ receiver (never sees images)

    def _receive(self, y: torch.Tensor,
                 side: ReceiverSideInfo) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Received symbols -> reconstructed features for both views, (B, S, D) each.
        Takes ReceiverSideInfo, never a view dict: no pixels reach this code path.
        """
        if self.n_user == 1:
            feat = self.decoders[0](y, token_dims=(2, side.n_tokens))   # (B, 2, S, D)
            return feat[:, 0], feat[:, 1]

        f1 = self.decoders[0](y, token_dims=(1, side.n_tokens))[:, 0]
        f2 = self.decoders[1](y, token_dims=(1, side.n_tokens))[:, 0]
        return f1, f2

    @torch.no_grad()
    def _rebuild_pos(self, true_shape: torch.Tensor,
                     img_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Rebuild the patch positions at the receiver from image geometry alone.

        Regenerating rather than forwarding `pos` from the encoder keeps the receiver
        free of any noise-free side channel. It has to reproduce the patch embedding's
        layout exactly, and the two DUSt3R variants differ:

        PatchEmbedDust3R.forward  ->  position_getter(B, x.size(2), x.size(3))
            the grid follows the image tensor; true_shape is not consulted at all.
        ManyAR_PatchEmbed.forward ->  portrait images get position_getter(1, W, H)
            i.e. a transposed grid, because the projection is applied to a transposed
            image so that a whole batch can mix orientations.

        Getting this wrong raises nothing: the token count is identical either way, and
        the only symptom is silently wrong RoPE rotations. Hence _check_rebuilt_pos().
        """
        patch_embed = self.dust3r.patch_embed
        ph, pw = patch_embed.patch_size
        H, W = img_hw[0] // ph, img_hw[1] // pw

        B = true_shape.shape[0]
        device = true_shape.device

        if not self.transposes_portrait:
            return patch_embed.position_getter(B, H, W, device)

        pos = torch.zeros((B, H * W, 2), dtype=torch.int64, device=device)
        height, width = true_shape.T
        is_landscape = (width >= height)
        if bool(is_landscape.any()):
            pos[is_landscape] = patch_embed.position_getter(1, H, W, device)
        if bool((~is_landscape).any()):
            pos[~is_landscape] = patch_embed.position_getter(1, W, H, device)
        return pos

    def _check_rebuilt_pos(self, pos1: torch.Tensor, pos2: torch.Tensor,
                           side: ReceiverSideInfo):
        """
        Assert the receiver can reproduce the encoder's patch positions exactly.

        Runs once, on the first forward. This is the one place where the receiver's
        "I can derive this from side information" claim could quietly be false, and a
        mismatch degrades results without ever raising, so it is worth a hard check.
        """
        for name, pos, true_shape in (('view1', pos1, side.true_shape1),
                                      ('view2', pos2, side.true_shape2)):
            rebuilt = self._rebuild_pos(true_shape, side.img_hw)
            if not torch.equal(rebuilt, pos):
                raise RuntimeError(
                    f'receiver-side patch positions do not match the encoder for {name}: '
                    f'patch_embed is {self.patch_embed_kind}, img_hw={side.img_hw}, '
                    f'true_shape={true_shape[0].tolist()}. The RoPE rotations would be '
                    f'wrong and nothing else would report it -- fix _rebuild_pos() to '
                    f'mirror {self.patch_embed_kind}.forward().')

    def _decode_geometry(self, feat1: torch.Tensor, feat2: torch.Tensor,
                         side: ReceiverSideInfo) -> Tuple[dict, dict]:
        """DUSt3R's cross-view decoder + prediction heads, on the RECEIVED features."""
        pos1 = self._rebuild_pos(side.true_shape1, side.img_hw)
        pos2 = self._rebuild_pos(side.true_shape2, side.img_hw)

        dec1, dec2 = self.dust3r._decoder(feat1, pos1, feat2, pos2)

        # heads run in fp32, as in AsymmetricCroCo3DStereo.forward
        with torch.amp.autocast('cuda', enabled=False):
            res1 = self.dust3r._downstream_head(1, [t.float() for t in dec1],
                                                side.true_shape1)
            res2 = self.dust3r._downstream_head(2, [t.float() for t in dec2],
                                                side.true_shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2

    def _decode_appearance(self, feat1: torch.Tensor, feat2: torch.Tensor,
                           side: ReceiverSideInfo) -> Tuple[torch.Tensor, torch.Tensor]:
        """Colours, decoded from the same received features. No extra bandwidth."""
        # true_shape is only needed to undo ManyAR_PatchEmbed's portrait transposition;
        # PatchEmbedDust3R never transposes, so passing it there would corrupt the layout
        ts1 = side.true_shape1 if self.transposes_portrait else None
        ts2 = side.true_shape2 if self.transposes_portrait else None
        return (self.rgb_head(feat1, side.img_hw, ts1),
                self.rgb_head(feat2, side.img_hw, ts2))

    # ------------------------------------------------------------------ forward

    def forward(self, view1: dict, view2: dict) -> Tuple[dict, dict]:
        """
        Same contract as AsymmetricCroCo3DStereo.forward.

        With an rgb_head attached, res1['rgb'] / res2['rgb'] carry the reconstructed
        images in DUSt3R's [-1, 1] colour space; the DUSt3R losses ignore extra keys.
        """
        # --- transmitter
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_images(view1, view2)
        signals = self._transmit(feat1, feat2)

        # --- channel
        y = self._through_channel(signals)

        # --- receiver: from here on, only side information is available
        side = ReceiverSideInfo(true_shape1=shape1, true_shape2=shape2,
                                img_hw=tuple(view1['img'].shape[-2:]),
                                n_tokens=feat1.shape[1])

        if self._verify_pos:
            # pos1/pos2 are used ONLY for this check, never in the receiver path
            self._check_rebuilt_pos(pos1, pos2, side)
            self._verify_pos = False

        feat1_hat, feat2_hat = self._receive(y, side)

        self._record_stats(feat1, feat2, feat1_hat, feat2_hat, signals, y)

        res1, res2 = self._decode_geometry(feat1_hat, feat2_hat, side)

        if self.rgb_head is not None:
            rgb1, rgb2 = self._decode_appearance(feat1_hat, feat2_hat, side)
            res1['rgb'], res2['rgb'] = rgb1, rgb2

        return res1, res2

    def feature_mse(self, view1: dict, view2: dict) -> torch.Tensor:
        """
        Auxiliary loss: how far the received features are from the transmitted ones.

        Useful as a warm-up term -- early in training the codec is random and the 3D
        loss gives it almost no signal. Anneal the weight to 0; the task loss is what
        you actually care about.
        """
        (shape1, shape2), (feat1, feat2), _ = self._encode_images(view1, view2)
        y = self._through_channel(self._transmit(feat1, feat2))
        side = ReceiverSideInfo(shape1, shape2, tuple(view1['img'].shape[-2:]),
                                feat1.shape[1])
        f1h, f2h = self._receive(y, side)
        return (f1h - feat1).pow(2).mean() + (f2h - feat2).pow(2).mean()

    # -------------------------------------------------------------- diagnostics

    @torch.no_grad()
    def _record_stats(self, feat1, feat2, feat1_hat, feat2_hat, signals, y):
        """
        Cheap scalars worth logging every step. Detached, no graph is kept.

        Also published to the module-level LATEST_STATS so the training criterion can
        pick them up: DUSt3R's criterion signature is (view1, view2, pred1, pred2) with
        no model argument, and threading a model reference through it is more fragile
        than a per-process global that only ever carries diagnostics.
        """
        self.last_stats = {
            'feat_mse': ((feat1_hat - feat1).pow(2).mean()
                         + (feat2_hat - feat2).pow(2).mean()).item() / 2,
            'feat_rel_err': ((feat1_hat - feat1).norm() / feat1.norm()).item(),
            'tx_power': signals[0].abs().pow(2).mean().item(),
            'rx_power': y.abs().pow(2).mean().item(),
            'symbols': y.shape[-1],
        }
        snr = getattr(self.channel, 'snr_db', None)
        if snr is not None:
            self.last_stats['snr_db'] = float(snr) if not torch.is_tensor(snr) \
                else float(snr.float().mean())

        LATEST_STATS.clear()
        LATEST_STATS.update(self.last_stats)
        return self.last_stats


def build_semcom_dust3r(dust3r: nn.Module,
                        channel: Channel,
                        n_user: int = 1,
                        bandwidth_ratio: float = 1 / 6,
                        share_codec: bool = True,
                        power_constraints: Optional[Sequence[float]] = None,
                        with_rgb_head: bool = False,
                        freeze: str = 'dust3r',
                        verify_pos: bool = True) -> SemComDust3R:
    """
    Convenience constructor that reads the feature dim and patch size off the DUSt3R
    model, so the codec can never silently disagree with the backbone.

    Args:
        share_codec: reuse one encoder/decoder pair for every user. Fine for n_user == 1
            and for a first two-user run; with two users it makes the superposition
            symmetric under swapping them, so no decoder can tell the users apart.
    """
    from .codec import make_codec_pair

    feat_dim = dust3r.enc_embed_dim if hasattr(dust3r, 'enc_embed_dim') \
        else dust3r.croco_args['enc_embed_dim']

    if share_codec:
        enc, dec = make_codec_pair(feat_dim, bandwidth_ratio)
        encoders, decoders = [enc] * n_user, [dec] * n_user
    else:
        pairs = [make_codec_pair(feat_dim, bandwidth_ratio) for _ in range(n_user)]
        encoders = [p[0] for p in pairs]
        decoders = [p[1] for p in pairs]

    head = RGBHead(feat_dim, dust3r.patch_embed.patch_size[0]) if with_rgb_head else None

    return SemComDust3R(dust3r, encoders, decoders, channel,
                        power_constraints=power_constraints, freeze=freeze,
                        rgb_head=head, verify_pos=verify_pos)


if __name__ == '__main__':
    import dust3r.utils.path_to_croco  # noqa: F401
    from dust3r.model import AsymmetricCroCo3DStereo

    from .channel import AWGNMultiUplinkChannel, AWGNSingleChannel, NopChannel

    torch.manual_seed(0)

    # RoPE2D is the cuda-compiled cuRoPE2D when croco/curope is built, and that kernel
    # only accepts cuda tensors -- so run on the GPU when there is one.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # a tiny DUSt3R so this is quick; the real one is 512-dpt / ViT-L.
    # pos_embed='RoPE100' is required: DUSt3R asserts enc_pos_embed is None, which only
    # holds for the RoPE variants (the croco default 'cosine' builds one).
    H, W, P, D = 96, 128, 16, 64
    net = AsymmetricCroCo3DStereo(
        output_mode='pts3d', head_type='linear', depth_mode=('exp', -float('inf'), float('inf')),
        conf_mode=('exp', 1, float('inf')), freeze='none', landscape_only=False,
        patch_embed_cls='ManyAR_PatchEmbed', img_size=(H, W), patch_size=P,
        pos_embed='RoPE100',
        enc_embed_dim=D, enc_depth=2, enc_num_heads=2,
        dec_embed_dim=D, dec_depth=2, dec_num_heads=2,
    ).to(device)
    S = (H // P) * (W // P)
    print(f'toy DUSt3R: img {H}x{W}, patch {P}, D={D}, S={S} tokens')

    B = 2
    def make_view():
        return dict(img=torch.randn(B, 3, H, W, device=device),
                    true_shape=torch.tensor([[H, W]] * B, device=device),
                    idx=list(range(B)), instance=[str(i) for i in range(B)])

    v1, v2 = make_view(), make_view()

    # ---- 1. single transmitter, both views in one signal
    model = build_semcom_dust3r(net, AWGNSingleChannel(10.0), n_user=1,
                                bandwidth_ratio=1 / 6, with_rgb_head=True).to(device)
    r1, r2 = model(v1, v2)
    assert r1['pts3d'].shape == (B, H, W, 3), r1['pts3d'].shape
    assert r2['pts3d_in_other_view'].shape == (B, H, W, 3)
    assert r1['rgb'].shape == (B, 3, H, W)
    print(f"[ok] n_user=1: pts3d {tuple(r1['pts3d'].shape)}, rgb {tuple(r1['rgb'].shape)}")
    print(f'     stats: {model.last_stats}')

    # only the codec (+ rgb head) should be trainable
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith('dust3r.') for n in trainable), sorted(trainable)[:3]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[ok] freeze="dust3r": {n_train:,} trainable / {n_total:,} total')

    # ---- 2. the receiver rebuilds the patch positions correctly, for BOTH patch
    #         embeddings and BOTH orientations. The released 512_dpt checkpoint uses
    #         PatchEmbedDust3R (no portrait transposition), the training configs use
    #         ManyAR_PatchEmbed (portrait grids are transposed) -- they disagree, and a
    #         mismatch is silent, so cover the matrix.
    for cls_name in ('ManyAR_PatchEmbed', 'PatchEmbedDust3R'):
        sub = AsymmetricCroCo3DStereo(
            output_mode='pts3d', head_type='linear',
            depth_mode=('exp', -float('inf'), float('inf')),
            conf_mode=('exp', 1, float('inf')), freeze='none', landscape_only=False,
            patch_embed_cls=cls_name, img_size=(H, W), patch_size=P, pos_embed='RoPE100',
            enc_embed_dim=D, enc_depth=2, enc_num_heads=2,
            dec_embed_dim=D, dec_depth=2, dec_num_heads=2,
        ).to(device)
        m = build_semcom_dust3r(sub, AWGNSingleChannel(10.0), n_user=1).to(device)
        for orient, ts in (('landscape', [H, W]), ('portrait', [W, H])):
            va = dict(img=torch.randn(B, 3, H, W, device=device),
                      true_shape=torch.tensor([ts] * B, device=device))
            vb = dict(img=torch.randn(B, 3, H, W, device=device),
                      true_shape=torch.tensor([ts] * B, device=device))
            _, _, (pa, _) = m._encode_images(va, vb)
            assert torch.equal(m._rebuild_pos(va['true_shape'], (H, W)), pa), \
                f'{cls_name} / {orient}'
        print(f'[ok] {cls_name}: positions rebuilt exactly (landscape and portrait)')

    # ---- 3. gradients reach the codec through the channel and DUSt3R's decoder
    r1, r2 = model(v1, v2)
    (r1['pts3d'].mean() + r2['pts3d_in_other_view'].mean() + r1['rgb'].mean()).backward()
    g = model.encoders[0].channel_encoder.model[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print('[ok] gradients reach the channel encoder through the whole stack')

    # ---- 4. two transmitters superimposed
    model2 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, 10.0), n_user=2,
                                 bandwidth_ratio=1 / 6, share_codec=True).to(device)
    r1, r2 = model2(v1, v2)
    assert r1['pts3d'].shape == (B, H, W, 3)
    print(f'[ok] n_user=2 superimposed: {model2.last_stats}')

    # unequal power allocation is just a constructor argument
    model3 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, [15.0, 5.0]), n_user=2,
                                 power_constraints=[1.5, 0.5],
                                 share_codec=False).to(device)
    model3(v1, v2)
    print(f'[ok] near-far + power allocation: {model3.last_stats}')

    # ---- 5. noiseless sanity: the pipeline must still run, with lower feature error
    clean = build_semcom_dust3r(net, NopChannel(), n_user=1).to(device)
    clean(v1, v2)
    noisy = build_semcom_dust3r(net, AWGNSingleChannel(0.0), n_user=1).to(device)
    noisy(v1, v2)
    print(f"[ok] feat_rel_err: noiseless {clean.last_stats['feat_rel_err']:.4f} "
          f"< 0 dB {noisy.last_stats['feat_rel_err']:.4f}")

    # ---- 6. drop-in compatible with dust3r's training loop helper
    from dust3r.inference import loss_of_one_batch
    out = loss_of_one_batch([make_view(), make_view()], model, None, device)
    assert set(out) >= {'view1', 'view2', 'pred1', 'pred2', 'loss'}
    print('[ok] works inside dust3r.inference.loss_of_one_batch')
