#!/usr/bin/env python3
"""
Phase C training: End-to-end joint training of DUSt3R + DeepJSCC.
==================================================================

Unlike Phase B (frozen backbone, feature-MSE loss), Phase C trains the
*full pipeline* jointly:

    ViT encoder → JSCC encoder → channel → JSCC decoder → ViT decoder → head

using DUSt3R's original task-level loss (ConfLoss + Regr3D), which requires
ground-truth depth maps and camera poses.  Any dataset that extends
``BaseStereoViewDataset`` (ScanNetpp, Co3d, MegaDepth, BlendedMVS …) works.

Freeze strategies
-----------------
encoder (default)
    Freeze patch_embed + enc_blocks (ViT encoder).
    Train only: ViT decoder, prediction heads, JSCC encoder/decoder.
    Cheaper, good starting point.  Recommended when data is limited.

none
    Unfreeze everything.  Full joint optimisation.
    Use smaller backbone_lr_scale (≤0.1) and larger accum_iter (≥4) to
    prevent catastrophic forgetting of the pre-trained backbone.

Learning rates
--------------
Two param groups are used:
    • backbone  lr = lr * backbone_lr_scale  (default 0.1×)
    • jscc      lr = lr

Usage examples
--------------
# Warm-start JSCC from Phase B, freeze ViT encoder, single dataset:
CUDA_VISIBLE_DEVICES=1 python train_semcom_phaseC.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --jscc_path    checkpoints/jscc_phaseB_awgn_k512.pth \\
    --dataset      "ScanNetpp(split='train', ROOT='data/scannetpp_processed', \\
                       resolution=512, aug_crop=16)" \\
    --freeze       encoder \\
    --channel      awgn \\
    --snr_range    0 20 \\
    --channel_dim  512 \\
    --epochs       50 \\
    --batch_size   4 \\
    --accum_iter   4 \\
    --lr           1e-4 \\
    --backbone_lr_scale 0.1 \\
    --amp \\
    --output_dir   checkpoints/phaseC_awgn_k512/

# Full joint training with mixed datasets (no JSCC warm-start):
CUDA_VISIBLE_DEVICES=1 python train_semcom_phaseC.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --dataset      "1000 @ ScanNetpp(split='train', ROOT='data/scannetpp_processed', \\
                       resolution=512, aug_crop=16) + \\
                    1000 @ Co3d(split='train', ROOT='data/co3d_processed', \\
                       mask_bg='rand', resolution=512, aug_crop=16)" \\
    --freeze       none \\
    --channel      awgn \\
    --snr_range    0 20 \\
    --epochs       100 \\
    --batch_size   2 \\
    --accum_iter   8 \\
    --lr           5e-5 \\
    --backbone_lr_scale 0.05 \\
    --amp \\
    --output_dir   checkpoints/phaseC_joint/
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

from dust3r.model_semcom import build_semcom_model_phaseC
from dust3r.datasets import get_data_loader
from dust3r.inference import loss_of_one_batch
from dust3r.losses import *  # noqa – required for criterion eval()
import dust3r.utils.path_to_croco  # noqa


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase C: end-to-end DUSt3R + DeepJSCC joint training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--weights', required=True,
                   help='DUSt3R backbone checkpoint (.pth).')
    p.add_argument('--jscc_path', default=None,
                   help='(optional) Phase B or Phase C checkpoint for JSCC '
                        'warm-start.  If omitted, JSCC is randomly initialised.')
    p.add_argument('--feat_dim', type=int, default=1024,
                   help='Encoder output dimension D.  Ignored when --jscc_path given.')
    p.add_argument('--channel_dim', type=int, default=512,
                   help='JSCC channel symbol dim k.  Ignored when --jscc_path given.')
    p.add_argument('--hidden_dim', type=int, default=None,
                   help='JSCC MLP hidden width H.  Defaults to feat_dim.')

    # ── Channel ───────────────────────────────────────────────────────────────
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'],
                   help='Physical channel model.')
    p.add_argument('--snr_db', type=float, default=10.0,
                   help='Fixed training SNR in dB.  Overridden by --snr_range.')
    p.add_argument('--snr_range', type=float, nargs=2, default=None,
                   metavar=('MIN', 'MAX'),
                   help='Uniform random SNR range [min, max] dB per batch.')

    # ── Backbone freeze strategy ──────────────────────────────────────────────
    p.add_argument('--freeze', default='encoder',
                   choices=['none', 'encoder'],
                   help='"encoder" freezes ViT patch_embed + enc_blocks; '
                        '"none" trains everything.')

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument('--dataset', required=True,
                   help='Dataset string passed to eval().  Must extend '
                        'BaseStereoViewDataset and provide depth + pose.  '
                        'Example: "ScanNetpp(split=\'train\', ROOT=\'data/scannetpp\', '
                        'resolution=512, aug_crop=16)"')
    p.add_argument('--num_workers', type=int, default=4)

    # ── Loss ──────────────────────────────────────────────────────────────────
    p.add_argument('--criterion',
                   default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                   help='DUSt3R-style loss criterion (eval string).')

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate for JSCC parameters.')
    p.add_argument('--backbone_lr_scale', type=float, default=0.1,
                   help='LR multiplier for backbone parameters (< 1 prevents '
                        'catastrophic forgetting).')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--accum_iter', type=int, default=1,
                   help='Gradient accumulation steps (effective batch = '
                        'batch_size × accum_iter).')
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--amp', action='store_true',
                   help='Use Automatic Mixed Precision (recommended).')

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument('--output_dir', required=True,
                   help='Directory for checkpoints and training log.')
    p.add_argument('--resume', default=None,
                   help='Phase C checkpoint to resume training from.')
    p.add_argument('--init_weights_from', default=None,
                   help='Phase C checkpoint to load full model weights from '
                        '(backbone + JSCC), without restoring optimizer state. '
                        'Useful for switching freeze strategy (e.g. encoder → none).')
    p.add_argument('--print_freq', type=int, default=20,
                   help='Log every N batches.')
    p.add_argument('--save_freq', type=int, default=1,
                   help='Save checkpoint every N epochs.')

    return p.parse_args()


# ── Optimiser helpers ─────────────────────────────────────────────────────────

def make_param_groups(model, lr: float, backbone_lr_scale: float):
    """
    Return AdamW param-groups with differential learning rates.

        • JSCC encoder + decoder  →  lr
        • Backbone (trainable)    →  lr × backbone_lr_scale
    """
    jscc_params = list(model.semcom.parameters())
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
    groups.append({
        'params': jscc_params,
        'lr':     lr,
        'name':   'jscc',
    })
    return groups


def cosine_schedule_fn(warmup: int, total: int):
    """Return a LambdaLR lambda for cosine annealing with linear warm-up."""
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
        'jscc_state_dict': model.semcom.state_dict(),   # backward compat
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
        # ── Sample SNR ────────────────────────────────────────────────────────
        snr_db = (random.uniform(*args.snr_range) if args.snr_range
                  else args.snr_db)
        model.semcom.snr_db = snr_db

        # ── Forward + task-level loss ─────────────────────────────────────────
        # loss_of_one_batch returns dict; result['loss'] = (scalar, details_dict)
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=False,  # keep it simple; dataset already has both views
            use_amp=args.amp,
        )
        loss_val, _details = result['loss']
        loss_val = loss_val / args.accum_iter

        # ── Backward ──────────────────────────────────────────────────────────
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

        # ── Logging ───────────────────────────────────────────────────────────
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

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 65)
    print('  DUSt3R × SemCom  —  Phase C: End-to-End Joint Training')
    print('=' * 65)
    print(f'  Device     : {device}')
    print(f'  Freeze     : {args.freeze}')
    print(f'  Channel    : {args.channel.upper()}')
    snr_info = (f'random [{args.snr_range[0]}, {args.snr_range[1]}] dB'
                if args.snr_range else f'fixed {args.snr_db} dB')
    print(f'  SNR        : {snr_info}')
    print(f'  Dataset    : {args.dataset[:80]}{"..." if len(args.dataset) > 80 else ""}')
    print(f'  Output     : {args.output_dir}')
    print('=' * 65 + '\n')

    # ── Build model ───────────────────────────────────────────────────────────
    model, jscc_config = build_semcom_model_phaseC(
        dust3r_path=args.weights,
        jscc_path=args.jscc_path,
        device=device,
        freeze=args.freeze,
        snr_db=args.snr_db,
        channel=args.channel,
        channel_dim=args.channel_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
    )

    # ── Optional: load full model weights (no optimizer state) ─────────────
    if args.init_weights_from and os.path.exists(args.init_weights_from):
        init_ckpt = torch.load(args.init_weights_from, map_location='cpu',
                               weights_only=False)
        model.load_state_dict(init_ckpt['model'])
        print(f'  [init_weights_from] Loaded full model weights from '
              f'{args.init_weights_from}  '
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
        model.semcom.snr_db = args.snr_db  # reset to current inference SNR
        print(f'  Resumed from epoch {start_epoch} ({args.resume})')

    # Full config for checkpoints
    cfg = {
        **jscc_config,
        'snr_db':               args.snr_range if args.snr_range else args.snr_db,
        'loss':                 args.criterion,
        'freeze':               args.freeze,
        'backbone_lr_scale':    args.backbone_lr_scale,
    }

    # Save config once
    with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
        json.dump({**cfg, 'epochs': args.epochs, 'batch_size': args.batch_size,
                   'lr': args.lr, 'accum_iter': args.accum_iter}, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\n  Starting training from epoch {start_epoch + 1} ...\n')

    for epoch in range(start_epoch, args.epochs):
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
