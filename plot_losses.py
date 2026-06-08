#!/usr/bin/env python3
"""
Training Loss Plotter
=====================
Plots per-epoch training loss from JSCC and/or E2E training runs.

Accepts two file types
  - JSON  files from train_jscc.py  →  {'epochs': [...], 'losses': [...]}
  - .pth  files from train_e2e.py   →  ckpt['train_losses'] (list)

Usage
-----
  # Single JSCC losses file
  python plot_losses.py checkpoints/jscc_awgn_k512_losses.json

  # Compare JSCC  vs  E2E  on same axes
  python plot_losses.py \\
      checkpoints/jscc_awgn_k512_losses.json \\
      checkpoints/phaseC_awgn_none_k512/checkpoint-last.pth

  # Save to directory instead of showing interactively
  python plot_losses.py ... --outdir figures/losses/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> tuple[list, list, dict]:
    """Load a JSCC losses JSON.  Returns (epochs, losses, meta)."""
    with open(path) as f:
        d = json.load(f)
    epochs = d['epochs']
    losses = d['losses']
    meta   = {k: v for k, v in d.items() if k not in ('epochs', 'losses')}
    return epochs, losses, meta


def _load_pth(path: str) -> tuple[list, list, dict]:
    """Load an E2E checkpoint .pth.  Returns (epochs, losses, meta)."""
    try:
        import torch
    except ImportError:
        sys.exit('[plot_losses] torch not available — cannot load .pth file')
    ckpt   = torch.load(path, map_location='cpu', weights_only=False)
    losses = ckpt.get('train_losses', [])
    epochs = list(range(len(losses)))
    meta   = ckpt.get('config', {})
    return epochs, losses, meta


def load_series(path: str) -> tuple[list, list, dict]:
    """Auto-dispatch based on file extension."""
    if path.endswith('.pth'):
        return _load_pth(path)
    else:
        return _load_json(path)


# ── Label ─────────────────────────────────────────────────────────────────────

def _auto_label(path: str, meta: dict) -> str:
    # E2E checkpoint directory name
    if path.endswith('.pth'):
        stem = os.path.basename(os.path.dirname(path))
        # e.g.  phaseC_awgn_none_k512  ->  E2E AWGN k=512 (freeze=none)
        channel = meta.get('channel', '').upper()
        cdim    = meta.get('channel_dim', '')
        freeze  = meta.get('freeze', '')
        parts   = ['E2E', channel]
        if cdim:
            parts.append(f'k={cdim}')
        if freeze:
            parts.append(f'freeze={freeze}')
        return ' '.join(p for p in parts if p)

    # JSCC losses JSON  (e.g. jscc_awgn_k512_losses.json)
    stem = os.path.basename(path)
    stem = stem.replace('_losses.json', '').replace('.json', '')
    for prefix in ('jscc_phaseB_', 'jscc_'):
        stem = stem.replace(prefix, '')
    # stem is now something like  awgn_k512  or  rayleigh_k512
    channel = meta.get('channel', '').upper()
    cdim    = meta.get('channel_dim', '')
    if not channel:
        # Try to guess from stem
        channel = 'AWGN' if 'awgn' in stem else ('RAYLEIGH' if 'rayleigh' in stem else '')
    if not cdim:
        import re
        m = re.search(r'k(\d+)', stem)
        cdim = m.group(1) if m else ''
    parts = ['JSCC', channel]
    if cdim:
        parts.append(f'k={cdim}')
    return ' '.join(p for p in parts if p)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_loss_plots(paths: list[str], outdir: str | None, smooth: int = 0):
    """
    Draw training-loss curves.

    When series have very different y-ranges (e.g. MSE loss vs ConfLoss),
    each series is plotted on its own twin y-axis with matching colour.

    Parameters
    ----------
    paths   : list of JSON or .pth file paths
    outdir  : if set, save PNG there; else show interactively
    smooth  : optional moving-average window size (0 = auto)
    """
    series = []
    for p in paths:
        epochs, losses, meta = load_series(p)
        label = _auto_label(p, meta)
        series.append((np.array(epochs, dtype=float),
                        np.array(losses, dtype=float),
                        label, meta, p))

    # Detect whether y-ranges are incompatible (ratio > 5×)
    means = [np.nanmean(np.abs(s[1])) for s in series]
    multi_scale = (len(means) > 1 and
                   max(means) / (min(means) + 1e-9) > 5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training Loss Curves — DUSt3R × SemCom',
                 fontsize=14, fontweight='bold')
    ax_raw, ax_smooth = axes

    # Colour cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = [p['color'] for p in prop_cycle]

    def _smooth_xy(x, y, window):
        if len(y) < 5:
            return x, y
        w = max(3, len(y) // 10) if window == 0 else window
        kernel = np.ones(w) / w
        y_sm = np.convolve(y, kernel, mode='valid')
        x_sm = x[w // 2: w // 2 + len(y_sm)]
        return x_sm, y_sm

    twin_axes_raw    = []
    twin_axes_smooth = []

    for i, (x, y, label, meta, path) in enumerate(series):
        colour = colours[i % len(colours)]
        x_sm, y_sm = _smooth_xy(x, y, smooth)

        if multi_scale and i > 0:
            ax_twin_r = ax_raw.twinx()
            ax_twin_s = ax_smooth.twinx()
            offset = 60 * (i - 1)
            ax_twin_r.spines['right'].set_position(('outward', offset))
            ax_twin_s.spines['right'].set_position(('outward', offset))
            ax_twin_r.plot(x,    y,    linewidth=1.5, alpha=0.85, color=colour, label=label)
            ax_twin_r.tick_params(axis='y', labelcolor=colour)
            ax_twin_r.set_ylabel('Loss (right)', color=colour, fontsize=9)
            ax_twin_s.plot(x_sm, y_sm, linewidth=2,   alpha=1.0,  color=colour, label=label)
            ax_twin_s.tick_params(axis='y', labelcolor=colour)
            twin_axes_raw.append(ax_twin_r)
            twin_axes_smooth.append(ax_twin_s)
        else:
            ax_raw.plot(x,    y,    linewidth=1.5, alpha=0.85, color=colour, label=label)
            ax_smooth.plot(x_sm, y_sm, linewidth=2, alpha=1.0, color=colour, label=label)

    for ax, title, twins in [
            (ax_raw,    'Raw loss per epoch',           twin_axes_raw),
            (ax_smooth, 'Smoothed loss (moving avg)',   twin_axes_smooth)]:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        # Merge legend handles from main axis + all twin axes
        handles, labels = ax.get_legend_handles_labels()
        for twin in twins:
            h, l = twin.get_legend_handles_labels()
            handles += h
            labels  += l
        ax.legend(handles, labels, fontsize=9)

    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, 'training_losses.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'[Saved] {out}')

        # Also save each panel separately (single-axis, no twinx for simplicity)
        for name, raw_mode in [('loss_raw', True), ('loss_smooth', False)]:
            fig_s, ax_s = plt.subplots(figsize=(7, 5))
            prop = plt.rcParams['axes.prop_cycle']
            cols = [p['color'] for p in prop]
            for i, (x, y, label, meta, _) in enumerate(series):
                c = cols[i % len(cols)]
                if raw_mode:
                    ax_s.plot(x, y, linewidth=1.5, alpha=0.85,
                              label=label, color=c)
                else:
                    xs, ys = _smooth_xy(x, y, smooth)
                    ax_s.plot(xs, ys, linewidth=2, label=label, color=c)
            ax_s.set_xlabel('Epoch')
            ax_s.set_ylabel('Loss')
            title = 'Raw loss per epoch' if raw_mode else 'Smoothed loss (moving avg)'
            ax_s.set_title(title, fontsize=12, fontweight='bold')
            ax_s.legend(fontsize=9)
            ax_s.grid(True, alpha=0.3)
            fig_s.tight_layout()
            p_out = os.path.join(outdir, f'{name}.png')
            fig_s.savefig(p_out, dpi=150, bbox_inches='tight')
            plt.close(fig_s)
            print(f'[Saved] {p_out}')
    else:
        plt.show()

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Plot DUSt3R SemCom training loss curves',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'files', nargs='+',
        help='Loss JSON file(s) or checkpoint .pth file(s)')
    parser.add_argument(
        '--outdir', default=None,
        help='Directory to save plots (default: show interactively)')
    parser.add_argument(
        '--smooth', type=int, default=0,
        help='Moving-average window size (0 = auto)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    make_loss_plots(args.files, args.outdir, args.smooth)
