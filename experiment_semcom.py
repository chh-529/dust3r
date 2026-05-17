#!/usr/bin/env python3
"""
Phase A SemCom Experiment: SNR Sweep for DUSt3R
================================================
Evaluates how AWGN / Rayleigh channel noise applied to encoder tokens
affects DUSt3R reconstruction quality and Global Alignment.

Usage
-----
# Quick test with the built-in Chateau stereo pair:
  CUDA_VISIBLE_DEVICES=1 python experiment_semcom.py \
      --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# Custom image folder (≥2 images of the same scene):
  CUDA_VISIBLE_DEVICES=1 python experiment_semcom.py \
      --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
      --images path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg

# Change SNR sweep and channel type:
  CUDA_VISIBLE_DEVICES=1 python experiment_semcom.py \
      --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
      --snr_list inf 20 15 10 5 0 -5 \
      --channel rayleigh
"""

import argparse
import copy
import math
import os
import sys

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from dust3r.model_semcom import load_semcom_model, load_semcom_model_phaseB
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ── Default test images ───────────────────────────────────────────────────────
_DEFAULT_IMAGES = [
    'croco/assets/Chateau1.png',
    'croco/assets/Chateau2.png',
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def mean_confidence(output):
    """Average confidence score across all pairwise predictions."""
    confs = []
    for key in ('pred1', 'pred2'):
        conf = output[key].get('conf')
        if conf is not None:
            confs.append(conf.float().mean().item())
    return float(np.mean(confs)) if confs else float('nan')


def pointmap_mse(output_noisy, output_clean):
    """
    Mean squared error between noisy and clean 3D point predictions.
    Uses pred1['pts3d'] (view1-frame points) for comparison.
    Ignores NaN/Inf values.
    """
    def _get_pts(out):
        pts = out['pred1'].get('pts3d')
        if pts is None:
            pts = out['pred1'].get('pts3d_in_other_view')
        return pts.float() if pts is not None else None

    pts_noisy = _get_pts(output_noisy)
    pts_clean = _get_pts(output_clean)
    if pts_noisy is None or pts_clean is None:
        return float('nan')

    diff = pts_noisy - pts_clean
    mask = torch.isfinite(diff).all(dim=-1)
    if mask.sum() == 0:
        return float('nan')
    return diff[mask].pow(2).mean().item()


def run_global_alignment(output, device, n_imgs, niter=300, schedule='linear', lr=0.01):
    """Run global alignment and return (scene, final_loss).

    Returns (None, float('nan')) when Global Alignment fails due to
    degenerate point clouds (e.g., caused by very low SNR channel noise).
    """
    mode = (GlobalAlignerMode.PointCloudOptimizer
            if n_imgs > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    final_loss = None
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        try:
            final_loss = scene.compute_global_alignment(
                init='mst', niter=niter, schedule=schedule, lr=lr)
            if hasattr(final_loss, 'item'):
                final_loss = final_loss.item()
            elif isinstance(final_loss, (list, tuple)):
                final_loss = float(final_loss[-1])
        except Exception as e:
            # Degenerate point cloud (e.g., cv2.solvePnPRansac fails at very
            # low SNR when features are too corrupted for PnP initialization).
            print(f'[GA FAILED: {type(e).__name__}: {e}]', end=' ')
            return None, float('nan')
    return scene, final_loss


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(args):
    device = args.device

    # Parse SNR list (support 'inf' string)
    snr_list = []
    for s in args.snr_list:
        snr_list.append(float('inf') if s.lower() == 'inf' else float(s))

    # ── Load images ───────────────────────────────────────────────────────────
    image_paths = args.images if args.images else _DEFAULT_IMAGES
    print(f'\n[Experiment] Images : {image_paths}')
    print(f'[Experiment] SNR sweep : {snr_list} dB')
    print(f'[Experiment] Channel   : {args.channel}')
    print(f'[Experiment] Device    : {device}\n')

    imgs = load_images(image_paths, size=args.image_size, verbose=False)
    if len(imgs) == 1:
        import copy as _copy
        imgs = [imgs[0], _copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    n_imgs = len(imgs)

    # ── Run clean baseline first (SNR=inf) ────────────────────────────────────
    print('─' * 60)
    print('  Running CLEAN BASELINE (SNR = inf) ...')
    model_clean = load_semcom_model(
        args.weights, device, snr_db=float('inf'), verbose=False)
    with torch.no_grad():
        output_clean = inference(pairs, model_clean, device, batch_size=1, verbose=False)
    _, ga_loss_clean = run_global_alignment(
        output_clean, device, n_imgs, niter=args.niter)
    conf_clean = mean_confidence(output_clean)
    del model_clean
    torch.cuda.empty_cache()

    print(f'  Baseline  |  mean_conf = {conf_clean:.4f}  |  '
          f'GA_loss = {ga_loss_clean if ga_loss_clean is not None else "N/A (PairViewer)"}')
    print('─' * 60)

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    results = []

    for snr_db in snr_list:
        snr_label = 'inf' if snr_db == float('inf') else f'{snr_db:.1f}'
        print(f'  SNR = {snr_label:>6} dB  |  loading model ...', end=' ', flush=True)

        if args.jscc_weights:
            model = load_semcom_model_phaseB(
                args.weights, args.jscc_weights, device,
                snr_db=snr_db, verbose=False)
        else:
            model = load_semcom_model(
                args.weights, device,
                snr_db=snr_db, channel=args.channel, verbose=False)

        with torch.no_grad():
            output = inference(pairs, model, device, batch_size=1, verbose=False)

        conf = mean_confidence(output)
        mse = pointmap_mse(output, output_clean)

        _, ga_loss = run_global_alignment(
            output, device, n_imgs, niter=args.niter)

        if ga_loss is None:
            ga_str = 'N/A'
        elif np.isnan(ga_loss):
            ga_str = 'FAILED'
        else:
            ga_str = f'{ga_loss:.6f}'
        print(f'mean_conf = {conf:.4f}  |  '
              f'pts3d_MSE = {mse:.6f}  |  '
              f'GA_loss = {ga_str}')

        results.append({
            'snr_db': snr_db,
            'mean_conf': conf,
            'pts3d_mse': mse,
            'ga_loss': ga_loss,
        })

        del model
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    phase_label = 'Phase B' if args.jscc_weights else 'Phase A'
    print('\n' + '=' * 60)
    print(f'  SUMMARY  ({args.channel.upper()} channel, {phase_label})')
    print('=' * 60)
    header = f'  {"SNR (dB)":>10}  {"mean_conf":>10}  {"pts3d_MSE":>12}  {"GA_loss":>12}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    print(f'  {"baseline":>10}  {conf_clean:>10.4f}  {"—":>12}  '
          f'{str(ga_loss_clean) if ga_loss_clean else "N/A (Pair)":>12}')
    for r in results:
        snr_str = 'inf' if r['snr_db'] == float('inf') else f'{r["snr_db"]:.1f}'
        ga = r['ga_loss']
        if ga is None:
            ga_str = 'N/A'
        elif np.isnan(ga):
            ga_str = 'FAILED'
        else:
            ga_str = f'{ga:.6f}'
        print(f'  {snr_str:>10}  {r["mean_conf"]:>10.4f}  '
              f'{r["pts3d_mse"]:>12.6f}  {ga_str:>12}')
    print('=' * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        import json
        out_data = {
            'channel': args.channel,
            'phase': 'B' if args.jscc_weights else 'A',
            'jscc_weights': args.jscc_weights,
            'images': image_paths,
            'baseline': {'mean_conf': conf_clean, 'ga_loss': ga_loss_clean},
            'sweep': results,
        }
        # Convert inf to string for JSON serialisation
        for r in out_data['sweep']:
            if r['snr_db'] == float('inf'):
                r['snr_db'] = 'inf'
        with open(args.output, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f'\n[Saved results to {args.output}]')


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Phase A SemCom SNR sweep for DUSt3R',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', required=True,
                        help='Path to DUSt3R checkpoint (.pth)')
    parser.add_argument('--images', nargs='+', default=None,
                        help='Image paths (default: croco/assets/Chateau1/2.png)')
    parser.add_argument('--snr_list', nargs='+',
                        default=['inf', '20', '15', '10', '5', '0'],
                        help='SNR values in dB to sweep. Use "inf" for noiseless.')
    parser.add_argument('--channel', choices=['awgn', 'rayleigh'], default='awgn',
                        help='Physical channel model')
    parser.add_argument('--image_size', type=int, default=512, choices=[224, 512])
    parser.add_argument('--niter', type=int, default=300,
                        help='Global alignment iterations (only used for ≥3 images)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None,
                        help='Save results to this JSON file')
    parser.add_argument('--jscc_weights', default=None,
                        help='(Phase B) Path to JSCC checkpoint produced by '
                             'train_semcom_phaseB.py.  When provided, the JSCC '
                             'encoder/decoder is loaded from this file instead '
                             'of using the Phase A identity mapping.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_experiment(args)
