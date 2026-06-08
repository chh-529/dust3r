#!/usr/bin/env python3
"""
End-to-end Joint Training of DUSt3R + DeepJSCC.
================================================

Trains the full pipeline jointly:

    ViT encoder → JSCC encoder → channel → JSCC decoder → ViT decoder → head

using DUSt3R's original task-level loss (ConfLoss + Regr3D), which requires
ground-truth depth maps and camera poses.  Any dataset extending
``BaseStereoViewDataset`` (ScanNetpp, Co3d, MegaDepth, BlendedMVS …) works.

All parameters are trainable.  Two AdamW param groups are used so the
backbone can be updated at a lower learning rate to prevent catastrophic
forgetting of the pre-trained weights:

    • backbone  lr = lr × backbone_lr_scale  (default 0.1×)
    • jscc      lr = lr

Compression ratio
-----------------
Use either ``--channel_dim k`` (explicit number of channel symbols) or
``--ratio r`` (fraction of feat_dim, e.g. 0.5 → k=512 when D=1024).
The two flags are mutually exclusive.  When ``--jscc_path`` is given,
both are ignored and the ratio is loaded from the checkpoint.

Usage examples
--------------
# Warm-start JSCC from train_jscc.py, ratio=1/2:
CUDA_VISIBLE_DEVICES=1 python train_e2e.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --jscc_path    checkpoints/jscc_awgn_k512.pth \\
    --dataset      "BlendedMVS(split='train', ROOT='data/blendedmvs_processed', \\
                       resolution=512, aug_crop=16)" \\
    --channel      awgn \\
    --snr_range    0 20 \\
    --ratio        0.5 \\
    --epochs       50 \\
    --batch_size   4 \\
    --accum_iter   4 \\
    --lr           1e-4 \\
    --backbone_lr_scale 0.1 \\
    --amp \\
    --output_dir   checkpoints/e2e_awgn_r0.5/

# From scratch, ratio=1/4:
CUDA_VISIBLE_DEVICES=1 python train_e2e.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --dataset      "BlendedMVS(split='train', ROOT='data/blendedmvs_processed', \\
                       resolution=512, aug_crop=16)" \\
    --channel      awgn \\
    --snr_range    0 20 \\
    --ratio        0.25 \\
    --epochs       50 \\
    --batch_size   2 \\
    --accum_iter   8 \\
    --lr           5e-5 \\
    --backbone_lr_scale 0.05 \\
    --amp \\
    --output_dir   checkpoints/e2e_awgn_r0.25/
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dust3r.model_semcom import build_semcom_model
from dust3r.datasets import get_data_loader
from dust3r.inference import loss_of_one_batch
from dust3r.losses import *  # noqa – required for criterion eval()
import dust3r.utils.path_to_croco  # noqa


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='End-to-end DUSt3R + DeepJSCC joint training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument('--weights', required=True,
                   help='DUSt3R backbone checkpoint (.pth).')
    p.add_argument('--jscc_path', default=None,
                   help='JSCC checkpoint for warm-start (from train_jscc.py). '
                        'If omitted, JSCC is randomly initialised.')
    p.add_argument('--feat_dim', type=int, default=1024,
                   help='Encoder output dimension D.  Ignored when --jscc_path given.')
    p.add_argument('--hidden_dim', type=int, default=None,
                   help='JSCC MLP hidden width H.  Defaults to feat_dim.')

    # Compression ratio (mutually exclusive)
    dim_group = p.add_mutually_exclusive_group()
    dim_group.add_argument('--channel_dim', type=int, default=None,
                           help='JSCC channel symbol dim k (e.g. 512). '
                                'Mutually exclusive with --ratio. '
                                'Ignored when --jscc_path given.')
    dim_group.add_argument('--ratio', type=float, default=None,
                           help='Compression ratio k/D (e.g. 0.5 → k=512 when D=1024). '
                                'Mutually exclusive with --channel_dim. '
                                'Ignored when --jscc_path given.')
    dim_group.add_argument('--no_jscc', action='store_true',
                           help='Identity channel (no JSCC compression): power-norm → '
                                'channel → inverse-scale on the FULL feature vector. '
                                'Trains the backbone with noise injection but no '
                                'compression — the clean SemCom ablation baseline.')

    # Channel
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'])
    p.add_argument('--snr_db', type=float, default=10.0,
                   help='Fixed training SNR in dB.  Overridden by --snr_range.')
    p.add_argument('--snr_range', type=float, nargs=2, default=None,
                   metavar=('MIN', 'MAX'),
                   help='Uniform random SNR range [min, max] dB per batch.')

    # Dataset
    p.add_argument('--dataset', required=True,
                   help='Dataset string passed to eval().  Must extend '
                        'BaseStereoViewDataset and provide depth + pose.')
    p.add_argument('--num_workers', type=int, default=4)

    # Loss
    p.add_argument('--criterion',
                   default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                   help='DUSt3R-style loss criterion (eval string).')

    # Training
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate for JSCC parameters.')
    p.add_argument('--backbone_lr_scale', type=float, default=0.1,
                   help='LR multiplier for backbone parameters.')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--accum_iter', type=int, default=1,
                   help='Gradient accumulation steps.')
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--amp', action='store_true',
                   help='Use Automatic Mixed Precision.')

    # I/O
    p.add_argument('--output_dir', default=None,
                   help='Directory for checkpoints and training log. '
                        'Auto-generated from parameters if omitted: '
                        'checkpoints/e2e_{channel}_r{ratio}_{snr}/')
    p.add_argument('--resume', default=None,
                   help='Checkpoint to resume training from.')
    p.add_argument('--init_weights_from', default=None,
                   help='Checkpoint to load full model weights from (no optimizer '
                        'state).  Useful for re-starting with a different config.')
    p.add_argument('--print_freq', type=int, default=20,
                   help='Log every N batches.')
    p.add_argument('--save_freq', type=int, default=1,
                   help='Save checkpoint every N epochs.')

    return p.parse_args()


# ── Optimiser helpers ─────────────────────────────────────────────────────────

def make_param_groups(model, lr: float, backbone_lr_scale: float):
    """Differential learning rates: JSCC at lr, backbone at lr × scale."""
    jscc_params = [p for p in model.semcom.parameters() if p.requires_grad]
    jscc_ids    = {id(p) for p in jscc_params}
    backbone_params = [
        p for p in model.parameters()
        if id(p) not in jscc_ids and p.requires_grad
    ]
    groups = []
    if backbone_params:
        groups.append({
            'params': backbone_params,
            'lr':     lr * backbone_lr_scale,
            'name':   'backbone',
        })
    # Identity mode (--no_jscc) has no JSCC parameters; skip the empty group.
    if jscc_params:
        groups.append({'params': jscc_params, 'lr': lr, 'name': 'jscc'})
    return groups


def cosine_schedule_fn(warmup: int, total: int):
    """LambdaLR lambda for cosine annealing with linear warm-up."""
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total - warmup)
        return max(0., 0.5 * (1 + math.cos(math.pi * t)))
    return fn


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                    train_losses, cfg, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    state = {
        'model':           model.state_dict(),
        'jscc_state_dict': model.semcom.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'scheduler':       scheduler.state_dict(),
        'scaler':          scaler.state_dict() if scaler is not None else None,
        'config':          cfg,
        'epoch':           epoch,
        'train_losses':    train_losses,
    }
    path = os.path.join(output_dir, 'checkpoint-last.pth')
    torch.save(state, path)
    print(f'  [Saved] {path}')


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer, scaler,
                    device, epoch, total_epochs, args):
    model.train()
    optimizer.zero_grad()

    losses_buf = []
    t0 = time.time()
    n_batches = len(data_loader)

    for step, batch in enumerate(data_loader):
        snr_db = (random.uniform(*args.snr_range) if args.snr_range
                  else args.snr_db)
        model.semcom.snr_db = snr_db

        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=False,
            use_amp=args.amp,
        )
        loss_val, _details = result['loss']
        loss_val = loss_val / args.accum_iter

        if args.amp:
            scaler.scale(loss_val).backward()
        else:
            loss_val.backward()

        if (step + 1) % args.accum_iter == 0:
            if args.amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.clip_grad,
            )
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        losses_buf.append(loss_val.item() * args.accum_iter)

        if step % args.print_freq == 0:
            lr_jscc = optimizer.param_groups[-1]['lr']
            elapsed = time.time() - t0
            print(
                f'  Epoch {epoch + 1}/{total_epochs} '
                f'[{step}/{n_batches}]  '
                f'loss={np.mean(losses_buf[-50:]):.4f}  '
                f'SNR={snr_db:.1f}dB  '
                f'lr_jscc={lr_jscc:.2e}  '
                f't={elapsed:.1f}s'
            )
            t0 = time.time()

    return float(np.mean(losses_buf))


# ── Main ──────────────────────────────────────────────────────────────────────

def _auto_output_dir(channel: str, channel_dim: int | None, feat_dim: int,
                     snr_range, snr_db: float, jscc_path: str | None,
                     no_jscc: bool = False) -> str:
    """Generate a descriptive output directory name from training parameters."""
    if no_jscc:
        ratio_str = 'identity'
    elif channel_dim is not None:
        ratio_str = f'r{channel_dim/feat_dim:.4g}'
    elif jscc_path is not None:
        ratio_str = 'r_from_ckpt'
    else:
        ratio_str = 'r_unknown'

    if snr_range is not None:
        snr_str = f'snr{int(snr_range[0])}-{int(snr_range[1])}'
    else:
        snr_str = f'snr{snr_db:.4g}'

    return f'checkpoints/e2e_{channel}_{snr_str}_{ratio_str}/'


def _resolve_channel_dim(feat_dim: int, channel_dim: int | None,
                         ratio: float | None, jscc_path: str | None) -> int | None:
    """Return concrete channel_dim k, or None if jscc_path is provided (loaded from ckpt)."""
    if jscc_path is not None:
        return None  # will be read from checkpoint
    if channel_dim is not None:
        return channel_dim
    if ratio is not None:
        k = max(1, round(feat_dim * ratio))
        print(f'  ratio={ratio} × feat_dim={feat_dim} → channel_dim={k}')
        return k
    raise ValueError('Specify --channel_dim or --ratio when --jscc_path is not given.')


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.no_jscc:
        channel_dim = None  # identity mode: no JSCC compression
    else:
        channel_dim = _resolve_channel_dim(
            args.feat_dim, args.channel_dim, args.ratio, args.jscc_path)

    if args.output_dir is None:
        args.output_dir = _auto_output_dir(
            args.channel, channel_dim, args.feat_dim,
            args.snr_range, args.snr_db, args.jscc_path, args.no_jscc)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.no_jscc:
        ratio_str = 'identity (no JSCC compression)'
    elif channel_dim:
        ratio_str = f'{channel_dim}/{args.feat_dim} = {channel_dim/args.feat_dim:.3f}'
    else:
        ratio_str = 'from checkpoint'

    print('\n' + '=' * 65)
    print('  DUSt3R × SemCom  —  End-to-End Joint Training')
    print('=' * 65)
    print(f'  Device     : {device}')
    print(f'  Channel    : {args.channel.upper()}')
    snr_info = (f'random [{args.snr_range[0]}, {args.snr_range[1]}] dB'
                if args.snr_range else f'fixed {args.snr_db} dB')
    print(f'  SNR        : {snr_info}')
    print(f'  Ratio      : {ratio_str}')
    print(f'  Dataset    : {args.dataset[:80]}{"..." if len(args.dataset) > 80 else ""}')
    print(f'  Output     : {args.output_dir}')
    print('=' * 65 + '\n')

    # ── Build model ───────────────────────────────────────────────────────────
    # Identity mode → channel_dim=None (SemComBlock builds an identity, param-free
    # block).  The `or feat_dim//2` fallback is only a placeholder for the
    # jscc_path case, where build_semcom_model ignores channel_dim anyway.
    build_channel_dim = None if args.no_jscc else (channel_dim or args.feat_dim // 2)
    model, jscc_config = build_semcom_model(
        dust3r_path=args.weights,
        jscc_path=args.jscc_path,
        device=device,
        freeze='none',
        snr_db=args.snr_db,
        channel=args.channel,
        channel_dim=build_channel_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
    )

    if args.init_weights_from and os.path.exists(args.init_weights_from):
        init_ckpt = torch.load(args.init_weights_from, map_location='cpu',
                               weights_only=False)
        model.load_state_dict(init_ckpt['model'])
        print(f'  [init_weights_from] Loaded from {args.init_weights_from}  '
              f'(epoch {init_ckpt.get("epoch", "?")})')
        del init_ckpt

    # ── Criterion ────────────────────────────────────────────────────────────
    criterion = eval(args.criterion)
    print(f'  Criterion  : {criterion}')

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    print('\n  Loading dataset ...')
    data_loader = get_data_loader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_mem=True,
    )
    print(f'  Batches/epoch: {len(data_loader)}  '
          f'(batch_size={args.batch_size}, accum_iter={args.accum_iter}  '
          f'→  effective batch={args.batch_size * args.accum_iter})\n')

    # ── Optimiser ─────────────────────────────────────────────────────────────
    param_groups = make_param_groups(model, args.lr, args.backbone_lr_scale)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosine_schedule_fn(args.warmup_epochs, args.epochs),
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    train_losses = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        train_losses = ckpt.get('train_losses', [])
        model.semcom.snr_db = args.snr_db
        print(f'  Resumed from epoch {start_epoch} ({args.resume})')

    cfg = {
        **jscc_config,
        'snr_db':            args.snr_range if args.snr_range else args.snr_db,
        'loss':              args.criterion,
        'backbone_lr_scale': args.backbone_lr_scale,
    }

    with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
        json.dump({**cfg, 'epochs': args.epochs, 'batch_size': args.batch_size,
                   'lr': args.lr, 'accum_iter': args.accum_iter}, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\n  Starting training from epoch {start_epoch + 1} ...\n')

    for epoch in range(start_epoch, args.epochs):
        if hasattr(data_loader.dataset, 'set_epoch'):
            data_loader.dataset.set_epoch(epoch)
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model, criterion, data_loader, optimizer, scaler,
            device, epoch, args.epochs, args,
        )
        train_losses.append(mean_loss)
        scheduler.step()

        lr_jscc = optimizer.param_groups[-1]['lr']
        print(f'\nEpoch {epoch + 1}/{args.epochs}  '
              f'mean_loss={mean_loss:.4f}  lr_jscc={lr_jscc:.2e}')

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                train_losses, cfg, args.output_dir,
            )

    print('\n  Training complete.')
    print(f'  Final checkpoint: {args.output_dir}/checkpoint-last.pth')


if __name__ == '__main__':
    main()
