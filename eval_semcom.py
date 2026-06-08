#!/usr/bin/env python3
"""
Dataset-level Evaluation for DUSt3R + SemCom
=============================================
Evaluates reconstruction quality across an SNR sweep using a held-out dataset
(default: BlendedMVS val split) against ground-truth depth + camera poses.

Metrics
-------
Primary (confidence-free, comparable across models):
  regr3d_l2 : Scale-normalized mean Euclidean pointmap error vs GT (avg_dis norm,
              view-1 camera frame).  This is Regr3D WITHOUT confidence weighting,
              so it is directly comparable across models — unlike task_loss.

Depth (standard monocular-depth metrics, view-1, median-scale aligned):
  abs_rel   : mean(|d_pred - d_gt| / d_gt)            — lower is better
  delta125  : fraction of pixels with max(d_p/d_g, d_g/d_p) < 1.25 — higher better

3D reconstruction (per-pair point cloud vs GT, avg_dis-normalized scale):
  acc       : Accuracy    — mean nearest-neighbour dist  pred → GT  (lower better)
  comp      : Completeness— mean nearest-neighbour dist  GT  → pred (lower better)
  chamfer   : 0.5 * (acc + comp)                                   (lower better)

Diagnostic (kept for backward compatibility, NOT comparable across models):
  task_loss : ConfLoss(Regr3D) — confidence-weighted; see confound note.
  mean_conf : Average model confidence.

Output JSON is compatible with plot_semcom.py for visualization.

Usage
-----
# Noise-only baseline, BlendedMVS val:
CUDA_VISIBLE_DEVICES=1 python eval_semcom.py \\
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --dataset "BlendedMVS(split='val', ROOT='data/blendedmvs_processed', \\
               resolution=512, aug_crop=16)" \\
    --output  results/eval_noisy_awgn.json

# With E2E checkpoint (from train_e2e.py):
CUDA_VISIBLE_DEVICES=1 python eval_semcom.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --jscc_weights checkpoints/e2e_awgn_snr0-20_r0.125/checkpoint-last.pth \\
    --dataset "BlendedMVS(split='val', ROOT='data/blendedmvs_processed', \\
               resolution=512, aug_crop=16)" \\
    --output  results/eval_e2e_awgn_r0.125.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dust3r.model_semcom import load_semcom_model
from dust3r.datasets import get_data_loader
from dust3r.inference import loss_of_one_batch
from dust3r.losses import *  # noqa — required for criterion eval()
from dust3r.utils.geometry import geotrf, inv
import dust3r.utils.path_to_croco  # noqa


# ── Metric helpers ──────────────────────────────────────────────────────────────

def _mean_conf(result: dict) -> float:
    """Average confidence from a loss_of_one_batch result dict."""
    confs = []
    for key in ('pred1', 'pred2'):
        conf = result[key].get('conf')
        if conf is not None:
            confs.append(conf.float().mean().item())
    return float(np.mean(confs)) if confs else float('nan')


def _subsample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Randomly subsample (P,3) point set down to at most n points."""
    if pts.shape[0] <= n:
        return pts
    idx = torch.randperm(pts.shape[0], device=pts.device)[:n]
    return pts[idx]


@torch.no_grad()
def _chamfer_acc_comp(pred_pts: torch.Tensor, gt_pts: torch.Tensor, n_points: int):
    """
    Accuracy / Completeness / Chamfer between two point sets (same scale).

    pred_pts, gt_pts : (P,3), (G,3) on the same device.
    Returns (acc, comp, chamfer) as Python floats (nan if a set is empty).
    """
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float('nan'), float('nan'), float('nan')
    pred_s = _subsample(pred_pts, n_points)
    gt_s   = _subsample(gt_pts,   n_points)
    d = torch.cdist(pred_s, gt_s)             # (p, g) pairwise Euclidean
    acc  = d.min(dim=1).values.mean().item()  # pred → nearest GT
    comp = d.min(dim=0).values.mean().item()  # GT   → nearest pred
    return acc, comp, 0.5 * (acc + comp)


@torch.no_grad()
def _depth_metrics(pred_d: torch.Tensor, gt_d: torch.Tensor, valid: torch.Tensor):
    """
    Median-scale-aligned monocular depth metrics on one image.

    pred_d, gt_d : (H,W) depth (z in camera frame).  valid : (H,W) bool mask.
    Returns (abs_rel, delta125) floats (nan if no valid pixels).
    """
    m = valid & (gt_d > 0) & (pred_d > 0)
    pv, gv = pred_d[m], gt_d[m]
    if pv.numel() == 0:
        return float('nan'), float('nan')
    # per-image median scale alignment (DUSt3R / monocular-depth convention)
    scale = gv.median() / pv.median().clamp(min=1e-8)
    pv = pv * scale
    abs_rel  = (torch.abs(pv - gv) / gv).mean().item()
    ratio    = torch.maximum(pv / gv, gv / pv)
    delta125 = (ratio < 1.25).float().mean().item()
    return abs_rel, delta125


@torch.no_grad()
def compute_metrics(result: dict, regr3d, n_points: int) -> dict:
    """
    Compute all reconstruction metrics for one batch.
    Returns a dict {metric_name: batch-mean float}.
    """
    gt1, gt2 = result['view1'], result['view2']
    pred1, pred2 = result['pred1'], result['pred2']

    # ── (1) Scale-normalized pointmaps (view-1 frame, avg_dis) via Regr3D ──────
    # Re-uses the exact transform the training loss uses, minus conf weighting.
    gt_p1, gt_p2, pr_p1, pr_p2, v1, v2, _ = regr3d.get_all_pts3d(gt1, gt2, pred1, pred2)

    # Un-weighted Regr3D L2 (mean Euclidean over valid points, both views)
    l1 = torch.norm(pr_p1[v1] - gt_p1[v1], dim=-1)
    l2 = torch.norm(pr_p2[v2] - gt_p2[v2], dim=-1)
    parts = [t for t in (l1, l2) if t.numel() > 0]
    regr3d_l2 = torch.cat(parts).mean().item() if parts else float('nan')

    # ── (2) Depth metrics on view-1 (raw scale, median-aligned per image) ─────
    in_cam1 = inv(gt1['camera_pose'])
    gt_d1   = geotrf(in_cam1, gt1['pts3d'])[..., 2]   # (B,H,W) GT depth in view-1 frame
    pred_d1 = pred1['pts3d'][..., 2]                  # (B,H,W) pred depth (view-1 local)
    vmask1  = gt1['valid_mask']

    # ── (3) Per-image depth + 3D reconstruction (clouds already same scale) ───
    abs_rels, deltas, accs, comps, chamfers = [], [], [], [], []
    B = gt_p1.shape[0]
    for b in range(B):
        ar, dl = _depth_metrics(pred_d1[b], gt_d1[b], vmask1[b])
        abs_rels.append(ar)
        deltas.append(dl)

        pred_cloud = torch.cat([pr_p1[b][v1[b]], pr_p2[b][v2[b]]], dim=0)
        gt_cloud   = torch.cat([gt_p1[b][v1[b]], gt_p2[b][v2[b]]], dim=0)
        a, c, ch = _chamfer_acc_comp(pred_cloud, gt_cloud, n_points)
        accs.append(a); comps.append(c); chamfers.append(ch)

    nanmean = lambda xs: float(np.nanmean(xs)) if len(xs) else float('nan')
    return {
        'regr3d_l2': regr3d_l2,
        'abs_rel':   nanmean(abs_rels),
        'delta125':  nanmean(deltas),
        'acc':       nanmean(accs),
        'comp':      nanmean(comps),
        'chamfer':   nanmean(chamfers),
        'mean_conf': _mean_conf(result),
    }


@torch.no_grad()
def eval_dataset(model, data_loader, criterion, regr3d, device,
                 snr_db: float, n_points: int):
    """Run model at ``snr_db`` across the full dataset; return mean metric dict."""
    model.eval()
    if model.semcom is not None:
        model.semcom.snr_db = snr_db

    snr_label = 'inf' if snr_db == float('inf') else f'{snr_db:.1f} dB'
    acc = {k: [] for k in ('task_loss', 'regr3d_l2', 'abs_rel', 'delta125',
                           'acc', 'comp', 'chamfer', 'mean_conf')}

    for batch in tqdm(data_loader, desc=f'  SNR={snr_label}', leave=False):
        result = loss_of_one_batch(batch, model, criterion, device, use_amp=False)
        loss_val, _ = result['loss']
        acc['task_loss'].append(loss_val.item())
        m = compute_metrics(result, regr3d, n_points)
        for k, v in m.items():
            acc[k].append(v)

    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Dataset-level SemCom evaluation for DUSt3R',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights', required=True,
                   help='DUSt3R backbone checkpoint (.pth)')
    p.add_argument('--jscc_weights', default=None,
                   help='JSCC or E2E checkpoint (from train_jscc.py / train_e2e.py). '
                        'Omit for noise-only (identity JSCC) mode.')
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'],
                   help='Physical channel model')
    p.add_argument('--snr_list', nargs='+',
                   default=['inf', '20', '15', '10', '5', '0'],
                   help='SNR values in dB to sweep. Use "inf" for noiseless.')
    p.add_argument('--dataset', required=True,
                   help='Dataset string passed to eval().  Must provide GT depth + pose. '
                        'Example: "BlendedMVS(split=\'val\', ROOT=\'data/blendedmvs\', '
                        'resolution=512, aug_crop=16)"')
    p.add_argument('--criterion',
                   default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                   help='DUSt3R-style loss criterion (eval string).')
    p.add_argument('--n_points', type=int, default=8192,
                   help='Points subsampled per image for Chamfer/Acc/Comp.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True,
                   help='Path to save the result JSON (compatible with plot_semcom.py)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    snr_list = [float('inf') if s.lower() == 'inf' else float(s)
                for s in args.snr_list]

    print('\n' + '=' * 65)
    print('  DUSt3R × SemCom — Dataset-level Evaluation')
    print('=' * 65)
    print(f'  Device     : {device}')
    print(f'  Channel    : {args.channel.upper()}')
    print(f'  SNR list   : {args.snr_list}')
    jscc_info = args.jscc_weights if args.jscc_weights else 'noise-only (no JSCC)'
    print(f'  JSCC       : {jscc_info}')
    print(f'  Dataset    : {args.dataset[:80]}{"..." if len(args.dataset) > 80 else ""}')
    print('=' * 65 + '\n')

    # Load model — use a finite SNR to ensure semcom block is created.
    init_snr = next((s for s in snr_list if s != float('inf')), 10.0)
    model = load_semcom_model(
        args.weights, device,
        snr_db=init_snr,
        channel=args.channel,
        jscc_path=args.jscc_weights,
        verbose=True,
    )

    criterion = eval(args.criterion)
    # Un-weighted Regr3D used to extract confidence-free pointmap metrics.
    regr3d = Regr3D(L21, norm_mode='avg_dis')
    print(f'\n  Criterion  : {criterion}')
    print(f'  Metrics    : regr3d_l2, abs_rel, delta125, acc, comp, chamfer '
          f'(+ task_loss, mean_conf)')

    print(f'\n  Loading dataset ...')
    data_loader = get_data_loader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )
    # ResizedDataset ("N @ ...") needs set_epoch() before indexing; harmless otherwise.
    if hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(0)
    n_pairs = len(data_loader.dataset)
    print(f'  {n_pairs} pairs, {len(data_loader)} batches\n')

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_results = []
    baseline = None

    metric_keys = ['regr3d_l2', 'abs_rel', 'delta125', 'acc', 'comp', 'chamfer',
                   'task_loss', 'mean_conf']

    for snr_db in snr_list:
        snr_label = 'inf' if snr_db == float('inf') else f'{snr_db:.1f}'
        print(f'  SNR = {snr_label:>6} dB', flush=True)

        m = eval_dataset(model, data_loader, criterion, regr3d, device,
                         snr_db, args.n_points)

        print(f'    regr3d_l2={m["regr3d_l2"]:.4f}  abs_rel={m["abs_rel"]:.4f}  '
              f'delta<1.25={m["delta125"]:.4f}')
        print(f'    chamfer={m["chamfer"]:.4f}  acc={m["acc"]:.4f}  comp={m["comp"]:.4f}')
        print(f'    task_loss={m["task_loss"]:.4f}  mean_conf={m["mean_conf"]:.4f}')

        if snr_db == float('inf'):
            baseline = dict(m)

        entry = {'snr_db': 'inf' if snr_db == float('inf') else snr_db}
        entry.update({k: m[k] for k in metric_keys})
        # legacy keys kept for plot_semcom.py compatibility
        entry['pts3d_mse'] = None
        entry['ga_loss']   = None
        sweep_results.append(entry)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    mode = 'jscc' if args.jscc_weights else 'noise-only'
    print(f'  SUMMARY  ({args.channel.upper()}, {mode})')
    print('=' * 78)
    hdr = f'  {"SNR":>5} {"regr3d_l2":>10} {"abs_rel":>9} {"d<1.25":>8} ' \
          f'{"chamfer":>9} {"task_loss":>10} {"conf":>8}'
    print(hdr)
    print('  ' + '-' * 74)
    for r in sweep_results:
        snr_str = str(r['snr_db'])
        print(f'  {snr_str:>5} {r["regr3d_l2"]:>10.4f} {r["abs_rel"]:>9.4f} '
              f'{r["delta125"]:>8.4f} {r["chamfer"]:>9.4f} '
              f'{r["task_loss"]:>10.4f} {r["mean_conf"]:>8.4f}')
    print('=' * 78)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = {
        'channel':      args.channel,
        'jscc_mode':    'jscc' if args.jscc_weights else 'noise-only',
        'jscc_weights': args.jscc_weights,
        'dataset':      args.dataset,
        'n_pairs':      n_pairs,
        'n_points':     args.n_points,
        'baseline':     baseline,
        'sweep':        sweep_results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  [Saved] {args.output}')


if __name__ == '__main__':
    main()
