#!/usr/bin/env python3
"""
JSCC Training for DUSt3R Semantic Communication.
=================================================

Trains only the JSCC encoder + decoder (SemComBlock with DeepJSCC).
The DUSt3R backbone (patch_embed, enc_blocks, decoder, heads) is
completely frozen throughout training.

Two loss modes
--------------
feat (default)
    MSE between clean encoder features and JSCC-reconstructed features.
    Fast; does not need ground-truth 3D labels.  Optimises for
    channel-robust feature recovery.

task
    End-to-end task distillation.  The clean DUSt3R acts as teacher;
    the JSCC model is the student.  Loss = MSE(pts3d_student, pts3d_teacher)
    + MSE(conf_student, conf_teacher).  Slower (two full forward passes)
    but directly aligned with the 3D reconstruction task.

Training strategy
-----------------
* Feature pre-computation cache: encoder features for every image are
  computed once and kept in GPU/CPU memory.  Each training step only
  runs the JSCC + (optionally) the decoder head.
* Random SNR training: supply --snr_range <min> <max> to sample the
  channel SNR uniformly per batch.  A fixed SNR is used when only
  --snr_db is given.

Usage examples
--------------
# Feature MSE loss, fixed SNR=10 dB, AWGN, channel_dim=512 (ratio 1/2)
CUDA_VISIBLE_DEVICES=1 python train_jscc.py \\
    --weights     checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --image_dirs  dust3r/images/my_desktop dust3r/images/timberland_boots \\
    --channel     awgn \\
    --snr_db      10 \\
    --channel_dim 512 \\
    --epochs      200 \\
    --lr          1e-3 \\
    --output      checkpoints/jscc_awgn_k512.pth

# Task distillation, random SNR in [0, 20] dB, Rayleigh, ratio 1/4
CUDA_VISIBLE_DEVICES=1 python train_jscc.py \\
    --weights     checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --image_dirs  dust3r/images/my_desktop dust3r/images/timberland_boots \\
    --channel     rayleigh \\
    --snr_range   0 20 \\
    --channel_dim 256 \\
    --loss        task \\
    --epochs      300 \\
    --output      checkpoints/jscc_rayleigh_k256.pth
"""

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from dust3r.model import load_model
from dust3r.semcom import SemComBlock
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_device(obj, device):
    """Recursively move tensors in a dict/list/tensor to *device*."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _load_images_from_dirs(image_dirs, image_size):
    """Scan *image_dirs* for PNG/JPG images and return a flat list of DUSt3R image dicts."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    all_imgs = []
    for d in image_dirs:
        paths = sorted(
            p for p in (os.path.join(d, f) for f in os.listdir(d))
            if os.path.splitext(p)[1].lower() in extensions
        )
        if not paths:
            print(f'  [warn] No images found in {d!r}')
            continue
        imgs = load_images(paths, size=image_size, verbose=False)
        base = len(all_imgs)
        for img in imgs:
            img['idx'] = base + img['idx']
        all_imgs.extend(imgs)
        print(f'  Loaded {len(imgs)} image(s) from {d!r}')
    return all_imgs


def _make_within_dir_pairs(image_dirs, all_imgs, image_size):
    """Return all (i, j) pairs where both images come from the same directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    dir_groups = []
    global_offset = 0
    for d in image_dirs:
        paths = sorted(
            p for p in (os.path.join(d, f) for f in os.listdir(d))
            if os.path.splitext(p)[1].lower() in extensions
        )
        n = len(paths)
        indices = list(range(global_offset, global_offset + n))
        if n >= 2:
            dir_groups.append(indices)
        global_offset += n

    pairs_idx = []
    for group in dir_groups:
        for i, j in itertools.combinations(group, 2):
            pairs_idx.append((i, j))
    return pairs_idx


# ── Feature cache ─────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_features(backbone, all_imgs, device):
    """
    Run the DUSt3R backbone encoder once for every image and cache
    (feat, pos, shape) on CPU.
    """
    backbone.eval()
    cache = {}
    print(f'  Pre-computing encoder features for {len(all_imgs)} image(s) ...')
    t0 = time.time()

    for img_dict in all_imgs:
        idx = img_dict['idx']
        true_shape = img_dict['true_shape']
        if not isinstance(true_shape, torch.Tensor):
            true_shape = torch.from_numpy(true_shape)
        view = {
            'img': img_dict['img'].to(device),
            'true_shape': true_shape.to(device),
            'idx': [idx],
            'instance': [img_dict['instance']],
        }
        (shape, _), (feat, _), (pos, _) = backbone._encode_symmetrized(view, view)
        cache[idx] = {
            'feat':  feat.cpu(),
            'pos':   pos.cpu(),
            'shape': shape,
        }

    elapsed = time.time() - t0
    feat_mb = sum(v['feat'].numel() * 4 for v in cache.values()) / 1e6
    print(f'  Done in {elapsed:.1f}s.  Cache size: {feat_mb:.1f} MB')
    return cache


# ── Task-distillation helpers ─────────────────────────────────────────────────

@torch.no_grad()
def run_decoder_head(backbone, feat1, pos1, shape1, feat2, pos2, shape2):
    """Run the frozen DUSt3R decoder + head on given features."""
    dec1, dec2 = backbone._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast('cuda', enabled=False):
        res1 = backbone._downstream_head(1, [t.float() for t in dec1], shape1)
        res2 = backbone._downstream_head(2, [t.float() for t in dec2], shape2)
    pts3d1 = res1['pts3d'].float()
    conf1  = res1['conf'].float()
    pts3d2 = res2.get('pts3d', res2.get('pts3d_in_other_view', None))
    if pts3d2 is not None:
        pts3d2 = pts3d2.float()
    conf2 = res2['conf'].float()
    return pts3d1, conf1, pts3d2, conf2


# ── Training loop ─────────────────────────────────────────────────────────────

def sample_snr(args):
    """Sample an SNR value according to the training strategy."""
    if args.snr_range is not None:
        lo, hi = args.snr_range
        return random.uniform(lo, hi)
    return args.snr_db


def train_one_epoch(backbone, semcom, optimizer, pairs_idx, feat_cache,
                    args, device, epoch):
    """Run one epoch of training.  Returns mean loss over all pairs."""
    semcom.train()
    losses = []
    random.shuffle(pairs_idx)

    for i_idx, j_idx in pairs_idx:
        cache_i = feat_cache[i_idx]
        cache_j = feat_cache[j_idx]

        feat_i_clean = cache_i['feat'].to(device)
        feat_j_clean = cache_j['feat'].to(device)
        pos_i   = cache_i['pos'].to(device)
        pos_j   = cache_j['pos'].to(device)
        shape_i = cache_i['shape']
        shape_j = cache_j['shape']

        snr_db = sample_snr(args)

        feat_i_hat = semcom(feat_i_clean, snr_db=snr_db)
        feat_j_hat = semcom(feat_j_clean, snr_db=snr_db)

        if args.loss == 'feat':
            loss = (
                F.mse_loss(feat_i_hat, feat_i_clean) +
                F.mse_loss(feat_j_hat, feat_j_clean)
            )
        else:  # 'task' — task distillation
            with torch.no_grad():
                pts3d_i_clean, conf_i_clean, pts3d_j_clean, conf_j_clean = \
                    run_decoder_head(backbone, feat_i_clean, pos_i, shape_i,
                                     feat_j_clean, pos_j, shape_j)

            pts3d_i_hat, conf_i_hat, pts3d_j_hat, conf_j_hat = \
                run_decoder_head(backbone, feat_i_hat, pos_i, shape_i,
                                 feat_j_hat, pos_j, shape_j)

            def pts_loss(pred, target):
                mask = torch.isfinite(target).all(dim=-1, keepdim=True)
                if mask.sum() == 0:
                    return pred.new_zeros(())
                return ((pred - target) * mask).pow(2).mean()

            loss_pts = pts_loss(pts3d_i_hat, pts3d_i_clean)
            if pts3d_j_hat is not None and pts3d_j_clean is not None:
                loss_pts = loss_pts + pts_loss(pts3d_j_hat, pts3d_j_clean)

            loss_conf = (
                F.l1_loss(conf_i_hat.log().clamp(-10, 10),
                          conf_i_clean.log().clamp(-10, 10)) +
                F.l1_loss(conf_j_hat.log().clamp(-10, 10),
                          conf_j_clean.log().clamp(-10, 10))
            )
            loss = loss_pts + args.task_conf_weight * loss_conf

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(semcom.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='JSCC training for DUSt3R SemCom (frozen backbone)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument('--weights', required=True,
                   help='DUSt3R checkpoint path.')
    p.add_argument('--feat_dim', type=int, default=1024,
                   help='Encoder output feature dimension D.')
    p.add_argument('--hidden_dim', type=int, default=None,
                   help='MLP hidden width H. Defaults to feat_dim.')

    # Compression ratio (mutually exclusive)
    dim_group = p.add_mutually_exclusive_group()
    dim_group.add_argument('--channel_dim', type=int, default=None,
                           help='JSCC channel symbol dimension k (e.g. 512). '
                                'Mutually exclusive with --ratio.')
    dim_group.add_argument('--ratio', type=float, default=None,
                           help='Compression ratio k/D (e.g. 0.5 → k=512 when D=1024). '
                                'Mutually exclusive with --channel_dim.')

    # Channel
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'],
                   help='Physical channel model.')
    p.add_argument('--snr_db', type=float, default=10.0,
                   help='Fixed training SNR in dB (used when --snr_range is absent).')
    p.add_argument('--snr_range', type=float, nargs=2, default=None,
                   metavar=('MIN', 'MAX'),
                   help='Random SNR range [min, max] dB per batch.  Overrides --snr_db.')

    # Data
    p.add_argument('--image_dirs', nargs='+', required=True,
                   help='Directories containing scene images.')
    p.add_argument('--image_size', type=int, default=512)

    # Training
    p.add_argument('--loss', default='feat', choices=['feat', 'task'],
                   help='"feat" = feature MSE (fast); "task" = task distillation.')
    p.add_argument('--task_conf_weight', type=float, default=0.1,
                   help='Weight of the confidence term in task loss.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # System
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_interval', type=int, default=10,
                   help='Print loss every N epochs.')
    p.add_argument('--output', required=True,
                   help='Output checkpoint path (.pth).')
    p.add_argument('--resume', default=None,
                   help='Resume from a previous JSCC checkpoint.')
    return p.parse_args()


def _resolve_channel_dim(feat_dim: int, channel_dim: int | None,
                         ratio: float | None) -> int:
    """Return concrete channel_dim k from --channel_dim or --ratio."""
    if channel_dim is not None:
        return channel_dim
    if ratio is not None:
        k = max(1, round(feat_dim * ratio))
        print(f'  ratio={ratio} × feat_dim={feat_dim} → channel_dim={k}')
        return k
    raise ValueError('Specify --channel_dim or --ratio.')


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    channel_dim = _resolve_channel_dim(args.feat_dim, args.channel_dim, args.ratio)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'\n[JSCC Training — frozen backbone]')
    print(f'  Device       : {device}')
    print(f'  Channel      : {args.channel}')
    if args.snr_range is not None:
        print(f'  SNR range    : [{args.snr_range[0]}, {args.snr_range[1]}] dB (random per batch)')
    else:
        print(f'  SNR          : {args.snr_db} dB (fixed)')
    print(f'  channel_dim  : {channel_dim}  (ratio {channel_dim/args.feat_dim:.3f})')
    _hidden = args.hidden_dim if args.hidden_dim is not None else args.feat_dim
    print(f'  hidden_dim   : {_hidden}')
    print(f'  Loss         : {args.loss}')
    print(f'  Epochs       : {args.epochs}  |  LR : {args.lr}')

    # ── Load frozen DUSt3R backbone ───────────────────────────────────────────
    print('\nLoading DUSt3R backbone (frozen) ...')
    backbone = load_model(args.weights, device=device, verbose=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ── Build SemComBlock (trainable) ─────────────────────────────────────────
    init_snr = args.snr_db if args.snr_range is None else float(np.mean(args.snr_range))
    semcom = SemComBlock(
        channel=args.channel,
        snr_db=init_snr,
        feat_dim=args.feat_dim,
        channel_dim=channel_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f'\nSemComBlock:\n{semcom}')
    n_params = sum(p.numel() for p in semcom.parameters())
    print(f'  Trainable parameters: {n_params:,}')

    start_epoch = 0
    train_losses = []

    if args.resume:
        print(f'\nResuming from {args.resume} ...')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        semcom.load_state_dict(ckpt['jscc_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        train_losses = ckpt.get('train_losses', [])
        print(f'  Resumed at epoch {start_epoch}')

    # ── Load images and build pair list ───────────────────────────────────────
    print('\nLoading images ...')
    all_imgs = _load_images_from_dirs(args.image_dirs, args.image_size)
    if len(all_imgs) < 2:
        raise RuntimeError(f'Need at least 2 images, got {len(all_imgs)}.')

    pairs_idx = _make_within_dir_pairs(args.image_dirs, all_imgs, args.image_size)
    if not pairs_idx:
        n = len(all_imgs)
        pairs_idx = [(all_imgs[i]['idx'], all_imgs[j]['idx'])
                     for i in range(n) for j in range(i + 1, n)]
    print(f'  Total images: {len(all_imgs)}  |  Training pairs: {len(pairs_idx)}')

    # ── Pre-compute encoder features ──────────────────────────────────────────
    print('\nPre-computing encoder features ...')
    feat_cache = precompute_features(backbone, all_imgs, device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        semcom.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    for _ in range(start_epoch):
        scheduler.step()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\nStarting training (epochs {start_epoch} → {args.epochs - 1}) ...\n')
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        mean_loss = train_one_epoch(
            backbone, semcom, optimizer, pairs_idx, feat_cache, args, device, epoch)
        scheduler.step()
        train_losses.append(mean_loss)

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(
                f'  epoch {epoch+1:4d}/{args.epochs}  '
                f'loss={mean_loss:.6f}  '
                f'lr={lr_now:.2e}  '
                f'elapsed={elapsed/60:.1f}min'
            )

    total_time = time.time() - t_start
    print(f'\nTraining complete in {total_time/60:.1f} min.')
    print(f'  Final loss: {train_losses[-1]:.6f}')

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    snr_info = args.snr_range if args.snr_range is not None else args.snr_db

    ckpt = {
        'jscc_state_dict': semcom.state_dict(),
        'config': {
            'feat_dim':    args.feat_dim,
            'channel_dim': channel_dim,
            'hidden_dim':  args.hidden_dim,
            'channel':     args.channel,
            'snr_db':      snr_info,
            'loss':        args.loss,
        },
        'epoch':        args.epochs - 1,
        'train_losses': train_losses,
    }
    torch.save(ckpt, args.output)
    print(f'Checkpoint saved → {args.output}')

    loss_json = args.output.replace('.pth', '_losses.json')
    with open(loss_json, 'w') as f:
        json.dump({'epochs': list(range(len(train_losses))),
                   'losses': train_losses}, f, indent=2)
    print(f'Loss curve saved → {loss_json}')


if __name__ == '__main__':
    main()
