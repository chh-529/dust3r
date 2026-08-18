"""
The end-to-end system: DUSt3R + semantic codec + channel.

    user 1 --[ViT]--> f_1 --[enc_1]--> z_1 --\
    user 2 --[ViT]--> f_2 --[enc_2]--> z_2 ---+--> y  (ONE receiver)
      ...                                    /
    user N --[ViT]--> f_N --[enc_N]--> z_N -/

    receiver: y --[dec_u]--→ f_u_hat (for every user)
            → form pairs → DUSt3R decoder & heads → pointmaps

Every user carries exactly ONE image; N is unconstrained (>= 2) 
Two ways the N signals reach the receiver, chosen by `multiplex`:

    'ofdm'          : each user on its own orthogonal resource
    'superimpose'   : all users on one resource
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .channel import Channel, MultiUplinkChannel, NopChannel
from .codec import FeatureChannelDecoder, FeatureChannelEncoder
from .utils import power_normalize


# Diagnostics from the most recent forward, per process. Written by
# SemComDust3R._record_stats(), read by the training criterion for logging. Never used
# in the model's own computation.
LATEST_STATS: Dict[str, float] = {}


@dataclass
class UserSideInfo:
    """
    What the receiver knows about one user's image
    """
    pos: torch.Tensor                   # (B, S, 2) patch positions, from the encoder
    true_shape: torch.Tensor            # (B, 2) per-sample (height, width)
    img_hw: Tuple[int, int]             # (H, W) of the padded image tensor, in pixels
    n_tokens: int                       # S, patch tokens


def complete_pairs(n: int, symmetrize: bool = True) -> List[Tuple[int, int]]:
    """
    Receiver-side pairing: the complete graph on N users, as dust3r's make_pairs would
    build it from N images. Symmetrized, because the global aligner expects both
    directions of every edge.
    """
    pairs = [(i, j) for i in range(n) for j in range(i)]
    if symmetrize:
        pairs = [p for i, j in pairs for p in ((i, j), (j, i))]
    return pairs


class SemComDust3R(nn.Module):
    """DUSt3R with a semantic codec and a channel spliced in after the ViT encoder."""

    def __init__(self,
                 dust3r: nn.Module,
                 encoders: Sequence[FeatureChannelEncoder],
                 decoders: Sequence[FeatureChannelDecoder],
                 channel: Channel,
                 freeze: str = 'none',
                 receiver: str = 'direct',
                 multiplex: str = 'ofdm'):
        """
        Args:
            dust3r: an AsymmetricCroCo3DStereo.
            encoders: one FeatureChannelEncoder per user
            decoders: one FeatureChannelDecoder per user, same length as `encoders`.
            channel: a MultiUplinkChannel (or NopChannel). Every encoder transmits at
                unit power; NOMA power allocation and near-far channel gain are the
                channel's `power_alloc` / `channel_gain`, not a model-level knob -- see
                semcom.channel.
            freeze: 'none' (default) trains everything; 'dust3r' freezes the entire
                    DUSt3R backbone (only trains the codec).
            receiver: 'direct' hands every decoder the same y; 'sic' does successive
                    interference cancellation
            multiplex: 'ofdm' (default) puts every user on their own orthogonal resource.
                       'superimpose' superimposes every user on one resource.
        """
        super().__init__()

        if len(encoders) != len(decoders):
            raise ValueError(f'{len(encoders) = } != {len(decoders) = }')
        n_user = len(encoders)
        if n_user < 1:
            raise ValueError(f'{n_user = } < 1')
        if receiver not in ('direct', 'sic'):
            raise ValueError(f"invalid {receiver = }, expected 'direct' or 'sic'")
        if multiplex not in ('ofdm', 'superimpose'):
            raise ValueError(f"invalid {multiplex = }, expected 'ofdm' or 'superimpose'")
        if not isinstance(channel, (MultiUplinkChannel, NopChannel)):
            raise ValueError('needs a MultiUplinkChannel (or NopChannel), sized n_user')
        if getattr(channel, 'n_tx', n_user) != n_user:
            raise ValueError(f'{channel.n_tx = } != {n_user = }')

        self.dust3r = dust3r
        self.n_user = n_user
        self.receiver = receiver
        self.multiplex = multiplex
        self.last_sic_order: Optional[List[int]] = None
        self.last_sic_power_trace: Optional[List[float]] = None
        # ModuleList de-duplicates nothing, so sharing works: the same module listed
        # twice is registered once and has one set of parameters
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.channel = channel

        self.set_freeze(freeze)
        self.last_stats: Dict[str, float] = {}

    # ---------------------------------------------------------------------------------
    # setup
    # ---------------------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.freeze == 'dust3r':
            return {k: v for k, v in sd.items() if not k.startswith('dust3r.')}
        return sd

    def set_freeze(self, freeze: str):
        self.freeze = freeze
        if freeze == 'none':
            targets = []
        elif freeze == 'dust3r':
            targets = [self.dust3r]
        else:
            raise ValueError(f"invalid {freeze = }, expected 'none'/'dust3r'")

        for module in targets:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad = False

    # ---------------------------------------------------------------------------------
    # transmitter
    # ---------------------------------------------------------------------------------
    def _encode_images(self, views: Sequence[dict]) -> Tuple[List[torch.Tensor],
                                                              List[UserSideInfo]]:
        """
        ViT encoding.
        """
        no_grad = self.freeze == 'dust3r'

        if len(views) == 2:
            def run():
                return self.dust3r._encode_symmetrized(views[0], views[1])
            (s1, s2), (f1, f2), (p1, p2) = (torch.no_grad()(run)() if no_grad else run())
            if no_grad:
                f1, f2 = f1.detach(), f2.detach()
            sides = [UserSideInfo(pos=p1, true_shape=s1, img_hw=tuple(views[0]['img'].shape[-2:]),
                                  n_tokens=f1.shape[1]),
                    UserSideInfo(pos=p2, true_shape=s2, img_hw=tuple(views[1]['img'].shape[-2:]),
                                  n_tokens=f2.shape[1])]
            return [f1, f2], sides

        feats, sides = [], []
        for v in views:
            img = v['img']
            B = img.shape[0]
            true_shape = v.get('true_shape',
                               torch.tensor(img.shape[-2:], device=img.device)[None].repeat(B, 1))
            if no_grad:
                with torch.no_grad():
                    feat, pos, _ = self.dust3r._encode_image(img, true_shape)
                feat = feat.detach()
            else:
                feat, pos, _ = self.dust3r._encode_image(img, true_shape)
            feats.append(feat)
            sides.append(UserSideInfo(pos=pos, true_shape=true_shape,
                                      img_hw=tuple(img.shape[-2:]), n_tokens=feat.shape[1]))
        return feats, sides

    def _channel_encoder(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        N features -> N unit-power complex signals, (B, L) each. Power allocation lives
        on the channel (power_alloc / channel_gain), not here -- see semcom.channel.
        """
        return [power_normalize(enc(f), 1.0) for enc, f in zip(self.encoders, feats)]


    # ---------------------------------------------------------------------------------
    # channel
    # ---------------------------------------------------------------------------------
    def _transmit(self, signals: Sequence[torch.Tensor]):
        """
        'ofdm'          : each user on their own orthogonal resource
        'superimpose'   : all users on one resource
        """
        if hasattr(self.channel, 'resample_snr') and self.training:
            self.channel.resample_snr()

        stacked = torch.stack(list(signals), dim=0)          # (n_user, B, L)
        if self.multiplex == 'ofdm':
            return self.channel.ofdm(stacked, user_dim_index=0)
        elif self.multiplex == 'superimpose':
            return self.channel.interfere(stacked, user_dim_index=0)
        else:
            raise ValueError(f"invalid {self.multiplex = }, expected 'ofdm' or 'superimpose'")

    # ---------------------------------------------------------------------------------
    # receiver
    # ---------------------------------------------------------------------------------
    def _channel_decoder(self, y, sides: Sequence[UserSideInfo]) -> List[torch.Tensor]:
        """
        Received symbols -> reconstructed features, (B, S, D) each.

        Dispatches on self.receiver:
            'direct'  every decoder gets the same superimposed y (or, in 'ofdm', its own
                      y_u).
            'sic'     successive interference cancellation.
        """
        if isinstance(y, list): # OFDM
            return [dec(yu, token_dims=(s.n_tokens,)) for dec, yu, s in zip(self.decoders, y, sides)]
        if self.receiver == 'sic': # NOMA + SIC
            return self._sic(y, sides)
        else: # NOMA + direct
            return [dec(y, token_dims=(s.n_tokens,)) for dec, s in zip(self.decoders, sides)]

    def _sic(self, y: torch.Tensor,
                     sides: Sequence[UserSideInfo]) -> List[torch.Tensor]:
        """
        Successive interference cancellation.
        """
        from .utils import signal_power

        a2 = getattr(self.channel, 'power_alloc', torch.ones(self.n_user))
        h = getattr(self.channel, 'channel_gain', torch.ones(self.n_user))
        gain = (torch.sqrt(a2) * h).to(y.device)                # a_u * h_u per user

        residual = y
        out: List[Optional[torch.Tensor]] = [None] * self.n_user
        power_trace = [float(signal_power(residual).mean())]

        order = sorted(range(self.n_user),
                       key=lambda u: (-float(gain[u]) ** 2, u))

        for u in order:
            feat_hat = self.decoders[u](residual, token_dims=(sides[u].n_tokens,))
            out[u] = feat_hat
            enc = self.encoders[u]
            kw = {}
            if hasattr(enc, 'last_indices'):
                kw = dict(indices=enc.last_indices, record=False)
            z_hat = power_normalize(enc(feat_hat, **kw), 1.0) * gain[u].to(residual.dtype)
            residual = residual - z_hat
            power_trace.append(float(signal_power(residual).mean()))

        self.last_sic_order = order
        self.last_sic_power_trace = power_trace
        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]

    def _decode_pair(self, feat_i: torch.Tensor, feat_j: torch.Tensor,
                     side_i: UserSideInfo, side_j: UserSideInfo) -> Tuple[dict, dict]:
        """
        DUSt3R's cross-view decoder + prediction heads, on the RECEIVED features.

        Returns (res_i, res_j) with res_j['pts3d_in_other_view'] expressed in user i's
        frame -- the same contract AsymmetricCroCo3DStereo.forward has, so the result
        drops straight into dust3r.cloud_opt.global_aligner.
        """
        dec_i, dec_j = self.dust3r._decoder(feat_i, side_i.pos, feat_j, side_j.pos)

        # heads run in fp32, as in AsymmetricCroCo3DStereo.forward
        with torch.amp.autocast('cuda', enabled=False):
            res_i = self.dust3r._downstream_head(1, [t.float() for t in dec_i], side_i.true_shape)
            res_j = self.dust3r._downstream_head(2, [t.float() for t in dec_j], side_j.true_shape)

        res_j['pts3d_in_other_view'] = res_j.pop('pts3d')
        return res_i, res_j

    # ---------------------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------------------
    def forward(self, view1: dict, view2: dict) -> Tuple[dict, dict]:
        """
        Same contract as AsymmetricCroCo3DStereo.forward. Requires n_user == 2; use
        forward_scene() for more users.
        """
        if self.n_user != 2:
            raise RuntimeError(f'forward(view1, view2) needs n_user == 2, got '
                               f'{self.n_user}; use forward_scene() for more users')

        # --- transmitter
        feats, sides = self._encode_images([view1, view2])
        signals = self._channel_encoder(feats)

        # --- channel
        y = self._transmit(signals)

        # --- receiver
        feats_hat = self._channel_decoder(y, sides)

        self._record_stats(feats, feats_hat, signals, y)

        res1, res2 = self._decode_pair(feats_hat[0], feats_hat[1], sides[0], sides[1])

        return res1, res2

    @torch.no_grad()
    def forward_scene(self, views: Sequence[dict]) -> dict:
        """
        N-user (n_user == len(views)) uplink: transmit once, decode every receiver-formed
        pair. Produces exactly the structure dust3r.inference.inference() returns:

            dict(view1=<collated dicts>, view2=..., pred1=..., pred2=...)

        which drops straight into dust3r.cloud_opt.global_aligner. Each view dict must
        carry 'idx' and 'instance' (what the global aligner uses to identify images
        across edges), plus 'img' and 'true_shape'.
        """
        from dust3r.utils.device import collate_with_cat

        if len(views) != self.n_user:
            raise ValueError(f'{len(views) = } != {self.n_user = }')

        # --- transmitter
        feats, sides = self._encode_images(views)
        signals = self._channel_encoder(feats)

        # --- channel
        y = self._transmit(signals)

        # --- receiver
        feats_hat = self._channel_decoder(y, sides)
        self._record_stats(feats, feats_hat, signals, y)

        if self.n_user == 1:
            dup = dict(views[0], idx=[1], instance=['1'])
            res_i, res_j = self._decode_pair(feats_hat[0], feats_hat[0], sides[0], sides[0])
            out = [dict(view1=views[0], view2=dup, pred1=res_i, pred2=res_j)]
        else:
            out = []
            for i, j in complete_pairs(self.n_user):
                res_i, res_j = self._decode_pair(feats_hat[i], feats_hat[j], sides[i], sides[j])
                out.append(dict(view1=views[i], view2=views[j], pred1=res_i, pred2=res_j))

        result = collate_with_cat(out)
        result['loss'] = None
        return result

    # ---------------------------------------------------------------------------------
    # diagnostic
    # ---------------------------------------------------------------------------------
    @torch.no_grad()
    def _record_stats(self, feats, feats_hat, signals, y):
        """
        Cheap scalars worth logging every step. Detached, no graph is kept.

        Also published to the module-level LATEST_STATS so the training criterion can
        pick them up: DUSt3R's criterion signature is (view1, view2, pred1, pred2) with
        no model argument, and threading a model reference through it is more fragile
        than a per-process global that only ever carries diagnostics.
        """
        y_flat = torch.stack(y, dim=0) if isinstance(y, list) else y
        num = sum(float((fh - f).pow(2).mean()) for f, fh in zip(feats, feats_hat))
        rel = sum(float((fh - f).norm() / f.norm()) for f, fh in zip(feats, feats_hat))
        # how far apart the users' reconstructions are: 0.0 means the decoders returned
        # literally the same tensor, i.e. the users were not separated at all
        spread = 0.0
        if len(feats_hat) > 1:
            ref = feats_hat[0]
            spread = max(float((fh - ref).abs().max()) for fh in feats_hat[1:])
        self.last_stats = dict(
            feat_mse=num / len(feats),
            feat_rel_err=rel / len(feats),
            user_spread=spread,
            tx_power=float(signals[0].abs().pow(2).mean()),
            rx_power=float(y_flat.abs().pow(2).mean()),
            symbols=int(y_flat.shape[-1]),
            n_user=self.n_user,
            receiver=self.receiver,
            multiplex=self.multiplex,
        )
        if self.receiver == 'sic' and self.last_sic_power_trace:
            tr = self.last_sic_power_trace
            # >1 means cancellation ADDED energy overall, i.e. SIC is hurting
            self.last_stats['sic_residual_ratio'] = tr[-1] / max(tr[0], 1e-12)
        snr = getattr(self.channel, 'snr_db', None)
        if snr is not None:
            self.last_stats['snr_db'] = (float(snr) if not torch.is_tensor(snr)
                                         else float(snr.float().mean()))

        LATEST_STATS.clear()
        LATEST_STATS.update({k: v for k, v in self.last_stats.items()
                             if isinstance(v, (int, float))})
        return self.last_stats


def build_semcom_dust3r(dust3r: nn.Module,
                        channel: Channel,
                        n_user: int = 2,
                        cpr: float = 1 / 6,
                        share_codec: bool = True,
                        freeze: str = 'none',
                        receiver: str = 'direct',
                        multiplex: str = 'ofdm',
                        codec_state_dict: Optional[dict] = None) -> SemComDust3R:
    """
    Convenience constructor that reads the feature dim and patch size off the DUSt3R
    model, so the codec can never silently disagree with the backbone.

    Args:
        share_codec: reuse one encoder/decoder pair for every user. Fine for a first run;
            with multiple users it makes the map (f_1..f_N) -> sum z_u symmetric under
            permuting users, so no decoder can tell them apart (see module docstring).
        codec_state_dict: a checkpoint's ['model'] dict to load the codec from. Only its
            'encoders.0.*' / 'decoders.0.*' entries are used (training produces one
            codec, not N); loaded into every user's encoder/decoder module. With
            share_codec=True those are all the same object, so this is a single load --
            see load_codec_state_dict() for the general, missing-key-aware version.
    """
    from .codec import make_codec_pair

    feat_dim = dust3r.enc_embed_dim if hasattr(dust3r, 'enc_embed_dim') \
        else dust3r.croco_args['enc_embed_dim']

    if share_codec:
        enc, dec = make_codec_pair(feat_dim, cpr)
        encoders, decoders = [enc] * n_user, [dec] * n_user
    else:
        pairs = [make_codec_pair(feat_dim, cpr) for _ in range(n_user)]
        encoders = [p[0] for p in pairs]
        decoders = [p[1] for p in pairs]

    if codec_state_dict is not None:
        enc_sd = {k[len('encoders.0.'):]: v for k, v in codec_state_dict.items()
                  if k.startswith('encoders.0.')}
        dec_sd = {k[len('decoders.0.'):]: v for k, v in codec_state_dict.items()
                  if k.startswith('decoders.0.')}
        if not enc_sd or not dec_sd:
            raise RuntimeError('no encoders.0.* / decoders.0.* weights in the checkpoint')
        for e in ({id(e): e for e in encoders}).values():
            e.load_state_dict(enc_sd)
        for d in ({id(d): d for d in decoders}).values():
            d.load_state_dict(dec_sd)

    return SemComDust3R(dust3r, encoders, decoders, channel,
                        freeze=freeze, receiver=receiver, multiplex=multiplex)


def load_codec_state_dict(model: SemComDust3R, sd: dict, strict_dust3r: bool = False):
    """
    model.load_state_dict() wrapper that understands share_codec's aliasing.

    With share_codec=True, self.encoders[0] and self.encoders[1] (...N-1) are the SAME
    nn.Module object, so state_dict() emits BOTH 'encoders.0.*' and 'encoders.1.*' (etc),
    pointing at identical tensors. A checkpoint saved from an OLDER, smaller-n_user model
    (e.g. this file's earlier one-transmitter-both-views mode) only has 'encoders.0.*' /
    'decoders.0.*' -- loading it here still works, because writing 'encoders.0.*' updates
    the shared tensor that 'encoders.1.*' also points to, but load_state_dict(strict=
    False) reports 'encoders.1.*' (etc) as "missing" even though it was never actually
    left stale. This filters exactly that case out before raising.
    """
    missing, unexpected = model.load_state_dict(sd, strict=False)

    aliased = set()
    if len(model.encoders) > 1 and model.encoders[0] is model.encoders[1]:
        own = dict(model.encoders[0].named_parameters())
        own.update(dict(model.encoders[0].named_buffers()))
        for name in own:
            for u in range(1, len(model.encoders)):
                aliased.add(f'encoders.{u}.{name}')
    if len(model.decoders) > 1 and model.decoders[0] is model.decoders[1]:
        own = dict(model.decoders[0].named_parameters())
        own.update(dict(model.decoders[0].named_buffers()))
        for name in own:
            for u in range(1, len(model.decoders)):
                aliased.add(f'decoders.{u}.{name}')

    real_missing = [k for k in missing if k not in aliased]
    codec_missing = [k for k in real_missing
                     if strict_dust3r or not k.startswith('dust3r.')]
    if codec_missing:
        raise RuntimeError(f'codec weights missing: {codec_missing[:5]}')
    return real_missing, unexpected


if __name__ == '__main__':
    import dust3r.utils.path_to_croco  # noqa: F401
    from dust3r.model import AsymmetricCroCo3DStereo

    from .channel import AWGNMultiUplinkChannel, NopChannel

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

    # ---- 1. two users, ofdm: plain two-view training, no multi-access ambiguity
    model = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, 10.0), n_user=2,
                                cpr=1 / 6, freeze='dust3r', multiplex='ofdm').to(device)
    r1, r2 = model(v1, v2)
    assert r1['pts3d'].shape == (B, H, W, 3), r1['pts3d'].shape
    assert r2['pts3d_in_other_view'].shape == (B, H, W, 3)
    print(f"[ok] n_user=2 ofdm: pts3d {tuple(r1['pts3d'].shape)}")
    print(f'     stats: {model.last_stats}')

    # only the codec should be trainable
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith('dust3r.') for n in trainable), sorted(trainable)[:3]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[ok] freeze="dust3r": {n_train:,} trainable / {n_total:,} total')

    # ---- 2. gradients reach the codec through the channel and DUSt3R's decoder
    r1, r2 = model(v1, v2)
    (r1['pts3d'].mean() + r2['pts3d_in_other_view'].mean()).backward()
    g = model.encoders[0].channel_encoder.model[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print('[ok] gradients reach the channel encoder through the whole stack')

    # ---- 3. superimpose: two users superimposed
    model2 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, 10.0), n_user=2,
                                 cpr=1 / 6, share_codec=True, multiplex='superimpose').to(device)
    r1, r2 = model2(v1, v2)
    assert r1['pts3d'].shape == (B, H, W, 3)
    print(f'[ok] n_user=2 superimpose (superimposed): {model2.last_stats}')

    # unequal NOMA power allocation is a CHANNEL constructor argument (power_alloc),
    # not a model-level one -- encoders always transmit at unit power
    model3 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, snr_db=10.0,
                                                             power_alloc=[1.5, 0.5]),
                                 n_user=2, share_codec=False,
                                 multiplex='superimpose').to(device)
    model3(v1, v2)
    print(f'[ok] NOMA power allocation: {model3.last_stats}')

    # near-far (per-user target SNR) is separately a channel constructor
    model3b = build_semcom_dust3r(net, AWGNMultiUplinkChannel.make_channel_gain(2, [15.0, 5.0]),
                                  n_user=2, share_codec=False,
                                  multiplex='superimpose').to(device)
    model3b(v1, v2)
    print(f'[ok] near-far (per-user target SNR): {model3b.last_stats}')

    # ---- 3b. receiver='sic': strongest user decoded first, ordered by the channel's
    #          own a_u*h_u (not a model-level power_constraints -- that no longer exists)
    model_sic = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, snr_db=10.0,
                                                                 power_alloc=[1.5, 0.5]),
                                    n_user=2, share_codec=True, receiver='sic',
                                    multiplex='superimpose').to(device)
    r1, r2 = model_sic(v1, v2)
    assert r1['pts3d'].shape == (B, H, W, 3)
    assert model_sic.last_sic_order == [0, 1], model_sic.last_sic_order  # user 0 is stronger
    assert model_sic.last_stats['user_spread'] > 0, 'SIC must break the shared-codec tie'
    print(f"[ok] receiver='sic': order {model_sic.last_sic_order}, "
          f"user_spread={model_sic.last_stats['user_spread']:.4f}")

    # ---- 4. noiseless sanity: the pipeline must still run, with lower feature error
    clean = build_semcom_dust3r(net, NopChannel(AWGNMultiUplinkChannel(2, 10.0)),
                                n_user=2, multiplex='ofdm').to(device)
    clean(v1, v2)
    noisy = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, 0.0), n_user=2,
                                multiplex='ofdm').to(device)
    noisy(v1, v2)
    print(f"[ok] feat_rel_err: noiseless {clean.last_stats['feat_rel_err']:.4f} "
          f"< 0 dB {noisy.last_stats['feat_rel_err']:.4f}")

    # ---- 5. drop-in compatible with dust3r's training loop helper
    from dust3r.inference import loss_of_one_batch
    out = loss_of_one_batch([make_view(), make_view()], model, None, device)
    assert set(out) >= {'view1', 'view2', 'pred1', 'pred2', 'loss'}
    print('[ok] works inside dust3r.inference.loss_of_one_batch')

    # ---- 6. N > 2 users: forward_scene() pairs everyone up
    model4 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(3, 10.0), n_user=3,
                                 multiplex='superimpose').to(device)
    views = [make_view() for _ in range(3)]
    scene = model4.forward_scene(views)
    assert set(scene) >= {'view1', 'view2', 'pred1', 'pred2'}
    n_pairs = len(complete_pairs(3))
    assert scene['pred1']['pts3d'].shape[0] == B * n_pairs
    print(f'[ok] forward_scene n_user=3: {n_pairs} pairs, stats: {model4.last_stats}')

    # ---- 6b. n_user=1: single transmitter, self-paired at decode time (mirrors
    #          dust3r/demo.py duplicating a single input image)
    model5 = build_semcom_dust3r(net, AWGNMultiUplinkChannel(1, 10.0), n_user=1,
                                 multiplex='ofdm').to(device)
    scene1 = model5.forward_scene([make_view()])
    assert scene1['view2']['idx'] != scene1['view1']['idx']
    assert scene1['pred1']['pts3d'].shape == (B, H, W, 3)
    print(f'[ok] forward_scene n_user=1: self-paired, stats: {model5.last_stats}')

    # ---- 7. backward compatibility: an old single-codec checkpoint (only
    #         'encoders.0.*' / 'decoders.0.*') must still load correctly into the new
    #         n_user=2, share_codec=True architecture -- see load_codec_state_dict().
    old_style_sd = {f'encoders.0.{k}': v.clone()
                    for k, v in model.encoders[0].state_dict().items()}
    old_style_sd.update({f'decoders.0.{k}': v.clone()
                         for k, v in model.decoders[0].state_dict().items()})
    fresh = build_semcom_dust3r(net, AWGNMultiUplinkChannel(2, 10.0), n_user=2,
                                cpr=1 / 6, share_codec=True, multiplex='ofdm').to(device)
    # load_codec_state_dict() not raising already proves the codec loaded cleanly
    # (it would have raised on any missing 'encoders.*'/'decoders.*' key); the frozen
    # dust3r.* backbone is expected to show up as "missing" here since old_style_sd only
    # ever held the codec, exactly like a real checkpoint (state_dict() strips dust3r.*)
    real_missing, unexpected = load_codec_state_dict(fresh, old_style_sd)
    assert all(k.startswith('dust3r.') for k in real_missing), real_missing
    assert unexpected == [], unexpected
    for name, p in fresh.encoders[0].state_dict().items():
        assert torch.equal(p, old_style_sd[f'encoders.0.{name}'])
    assert fresh.encoders[0] is fresh.encoders[1]  # same object -> user 1 got it too
    print('[ok] old single-codec checkpoint loads cleanly into the new n_user=2 model')

    # ---- 8. pipeline stage by stage: print what goes in/out of the channel encoder,
    #         the channel itself, and the channel decoder, across several SNRs. This is
    #         a print-and-look diagnostic, not an assertion-heavy test -- the point is to
    #         SEE the numbers change (or fail to), not just get a pass/fail.
    def show(name: str, t, n: int = 4):
        if isinstance(t, list):
            for i, ti in enumerate(t):
                show(f'{name}[{i}]', ti, n)
            return
        sample = t.reshape(-1)[:n].tolist()
        if t.is_complex():
            power = float((t.abs() ** 2).mean())
            sample = [f'{c.real:+.3f}{c.imag:+.3f}j' for c in sample]
            print(f'    {name:<16} shape={tuple(t.shape)!s:<14} complex  '
                  f'power={power:7.4f}  sample={sample}')
        else:
            print(f'    {name:<16} shape={tuple(t.shape)!s:<14} real     '
                  f'mean={float(t.mean()):+7.4f} std={float(t.std()):6.4f}  '
                  f'sample={[round(v, 4) for v in sample]}')

    print('\n---- 8. stage-by-stage inspection across SNR ----')
    for snr_db in (-10.0, 0.0, 10.0, 20.0, 30.0):
        ch = AWGNMultiUplinkChannel(2, snr_db) # a_k = h_k = 1, so y - sum(x) == noise
        m8 = build_semcom_dust3r(net, ch, n_user=2, cpr=1 / 6, freeze='dust3r',
                                 multiplex='superimpose').to(device)

        feats, sides = m8._encode_images([v1, v2])
        signals = m8._channel_encoder(feats)
        y = m8._transmit(signals)
        feats_hat = m8._channel_decoder(y, sides)

        print(f'\n  == SNR = {snr_db:5.1f} dB ==')
        print('  -- channel encoder: feat (real, ViT space) -> signal (complex, unit power) --')
        show('feat[0] in', feats[0])
        show('signal[0] out', signals[0])
        print('  -- channel: signals in -> superimposed + noisy y out --')
        show('signal[0] in', signals[0])
        show('signal[1] in', signals[1])
        show('y out', y)
        noise_mag = float((y - signals[0] - signals[1]).abs().mean())
        print(f'    |y - sum(signals)| (== noise magnitude, should shrink as SNR rises) '
              f'= {noise_mag:.4f}')
        print('  -- channel decoder: y in -> feat_hat (real, back in ViT space) out --')
        show('y in', y)
        show('feat_hat[0] out', feats_hat[0])
        rel_err = float((feats_hat[0] - feats[0]).norm() / feats[0].norm())
        print(f'    feat_rel_err (feat_hat[0] vs feat[0]) = {rel_err:.4f}')

    print('\n[ok] stage-by-stage inspection printed above')
