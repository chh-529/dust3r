#!/usr/bin/env python3
"""
Visualize DUSt3R + SemCom reconstruction quality.

For each requested SNR level, runs inference + global alignment and saves:
  - A side-by-side comparison PNG  (RGB | Depth | Confidence) per view
  - A 3D mesh GLB file

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python visualize_reconstruction.py \\
      --weights    checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
      --jscc_weights checkpoints/jscc_awgn_k512.pth \\
      --images  dust3r/images/my_desktop/1.png dust3r/images/my_desktop/2.png \\
                dust3r/images/my_desktop/3.png dust3r/images/my_desktop/4.png \\
                dust3r/images/my_desktop/5.png \\
      --snr_list inf 20 10 0 \\
      --scene_name desktop \\
      --outdir figures/reconstruction/
"""

import argparse
import copy
import math
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.model_semcom import load_semcom_model


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args, device: str, snr_db: float):
    """Load unified SemCom model at the given SNR."""
    return load_semcom_model(
        args.weights, device,
        snr_db=snr_db, channel=args.channel,
        jscc_path=args.jscc_weights, verbose=False,
    )


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct(model, imgs, device):
    """Run inference + global alignment. Returns scene object."""
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = (GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=False)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        try:
            scene.compute_global_alignment(
                init='mst', niter=300, schedule='linear', lr=0.01)
        except Exception as e:
            print(f'  [GA warning] {e}')

    return scene


# ── Visualization helpers ─────────────────────────────────────────────────────

CMAP = plt.get_cmap('jet')


def _norm(arr):
    """Normalize array to [0, 1] for colormap application."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def save_comparison_png(snr_labels, scenes, outpath, scene_name, channel, mode_label):
    """
    Save comparison grid: each column = one SNR level.
    Each column has 3 sub-columns: RGB | Depth | Confidence.
    Each row is a view.
    """
    n_snr = len(snr_labels)
    n_views = len(scenes[0].imgs)

    # Extract numpy arrays for all SNR settings
    data = []
    for scene in scenes:
        rgbs  = scene.imgs                               # list[ndarray H×W×3]
        deps  = to_numpy(scene.get_depthmaps())          # list[ndarray H×W]
        confs = to_numpy([c for c in scene.im_conf])     # list[ndarray H×W]
        data.append((rgbs, deps, confs))

    # Figure layout: rows = views, cols = n_snr × 3
    n_cols = n_snr * 3
    fig, axes = plt.subplots(
        n_views, n_cols,
        figsize=(n_cols * 2.8, n_views * 2.6),
        squeeze=False,
    )
    fig.suptitle(
        f'{scene_name} — {mode_label} ({channel.upper()}) '
        f'Reconstruction Quality',
        fontsize=13, fontweight='bold', y=1.01,
    )

    sub_titles = ['RGB', 'Depth', 'Conf']
    for si, snr_lbl in enumerate(snr_labels):
        for k, st in enumerate(sub_titles):
            label = 'clean' if snr_lbl == 'inf' else f'{snr_lbl} dB'
            axes[0, si * 3 + k].set_title(
                f'SNR={label}\n{st}', fontsize=8, pad=3)

    for vi in range(n_views):
        for si, (snr_lbl, (rgbs, deps, confs)) in enumerate(zip(snr_labels, data)):
            c = si * 3
            axes[vi, c  ].imshow(np.clip(rgbs[vi], 0, 1))
            axes[vi, c+1].imshow(CMAP(_norm(deps[vi])))
            axes[vi, c+2].imshow(CMAP(_norm(confs[vi])))
            for col in range(c, c + 3):
                axes[vi, col].axis('off')

        axes[vi, 0].set_ylabel(f'View {vi+1}', fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {outpath}')


def save_single_snr_png(scene, snr_label, outpath, scene_name, channel, mode_label):
    """
    Save per-SNR detail view: each row = one view,
    columns = [RGB, Depth, Confidence].
    """
    rgbs  = scene.imgs
    deps  = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    n_views = len(rgbs)

    fig, axes = plt.subplots(n_views, 3, figsize=(10, n_views * 3), squeeze=False)
    label = 'clean (no channel)' if snr_label == 'inf' else f'SNR = {snr_label} dB'
    fig.suptitle(
        f'{scene_name} — {label} ({mode_label} {channel.upper()})',
        fontsize=12, fontweight='bold',
    )
    for vi in range(n_views):
        axes[vi, 0].imshow(np.clip(rgbs[vi], 0, 1))
        axes[vi, 0].set_ylabel(f'View {vi+1}', fontsize=8)
        axes[vi, 1].imshow(CMAP(_norm(deps[vi])))
        axes[vi, 2].imshow(CMAP(_norm(confs[vi])))
        axes[vi, 0].set_title('RGB' if vi == 0 else '')
        axes[vi, 1].set_title('Depth' if vi == 0 else '')
        axes[vi, 2].set_title('Confidence' if vi == 0 else '')
        for col in range(3):
            axes[vi, col].axis('off')

    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {outpath}')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize DUSt3R + SemCom reconstruction quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights', required=True,
                   help='DUSt3R backbone checkpoint (.pth).')
    p.add_argument('--jscc_weights', default=None,
                   help='JSCC checkpoint (from train_jscc.py or train_e2e.py). '
                        'Omit for noise-only (identity JSCC) mode.')
    p.add_argument('--images', nargs='+', required=True,
                   help='Input image paths (≥2 images of the same scene).')
    p.add_argument('--snr_list', nargs='+', default=['inf', '20', '10', '0'],
                   help='SNR values in dB to evaluate. Use "inf" for clean baseline.')
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'],
                   help='Channel type (must match the trained checkpoint).')
    p.add_argument('--scene_name', default='scene',
                   help='Scene name for figure titles and file names.')
    p.add_argument('--outdir', default='figures/reconstruction/',
                   help='Output directory for PNGs and GLB files.')
    p.add_argument('--save_glb', action='store_true', default=True,
                   help='Also export 3D mesh as .glb for each SNR.')
    p.add_argument('--no_save_glb', dest='save_glb', action='store_false')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = args.device
    scene_name = args.scene_name
    mode_label = 'JSCC' if args.jscc_weights else 'noise-only'

    # Parse SNR list
    snr_values = []
    for s in args.snr_list:
        snr_values.append(('inf', float('inf')) if s == 'inf' else (s, float(s)))

    # Load images once (shared across SNR runs)
    print(f'Loading {len(args.images)} images ...')
    imgs_raw = load_images(args.images, size=512, verbose=False)
    if len(imgs_raw) == 1:
        imgs_raw = [imgs_raw[0], copy.deepcopy(imgs_raw[0])]
        imgs_raw[1]['idx'] = 1

    scenes = []
    snr_labels = []

    for snr_label, snr_db in snr_values:
        label_str = 'inf' if math.isinf(snr_db) else f'{snr_db:.0f}dB'
        print(f'\n[SNR={snr_label}] Loading model ...')
        model = load_model(args, device, snr_db)
        model.eval()

        print(f'[SNR={snr_label}] Running inference + global alignment ...')
        scene = reconstruct(model, imgs_raw, device)
        scenes.append(scene)
        snr_labels.append(snr_label)

        # Per-SNR detail PNG
        detail_path = os.path.join(
            args.outdir,
            f'{scene_name}_{mode_label}_{args.channel}_snr{snr_label}.png',
        )
        save_single_snr_png(scene, snr_label, detail_path, scene_name,
                            args.channel, mode_label)

        # 3D mesh GLB
        if args.save_glb:
            with tempfile.TemporaryDirectory() as tmpdir:
                glb_src = get_3D_model_from_scene(
                    tmpdir, silent=True, scene=scene,
                    min_conf_thr=3, as_pointcloud=False,
                    mask_sky=False, clean_depth=True,
                    transparent_cams=False, cam_size=0.05,
                )
                glb_dst = os.path.join(
                    args.outdir,
                    f'{scene_name}_{mode_label}_{args.channel}_snr{snr_label}.glb',
                )
                import shutil
                shutil.copy(glb_src, glb_dst)
                print(f'[Saved] {glb_dst}')

        del model
        torch.cuda.empty_cache()

    # Comparison grid across all SNR levels
    if len(scenes) > 1:
        grid_path = os.path.join(
            args.outdir,
            f'{scene_name}_{mode_label}_{args.channel}_comparison.png',
        )
        save_comparison_png(snr_labels, scenes, grid_path, scene_name,
                            args.channel, mode_label)

    print(f'\nAll outputs saved to: {args.outdir}')
