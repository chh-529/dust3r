#!/usr/bin/env python3
"""
Phase A SemCom Results Plotter
==============================
Reads one or more JSON result files produced by experiment_semcom.py
and generates publication-quality plots.

Usage
-----
# Single result file:
  python plot_semcom.py results_semcom_phaseA.json

# Compare multiple experiments (e.g., AWGN vs Rayleigh):
  python plot_semcom.py results_awgn.json results_rayleigh.json

# Save plots to a directory instead of showing interactively:
  python plot_semcom.py results_semcom_phaseA.json --outdir figures/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_sweep(result: dict):
    """
    Extract SNR, mean_conf, pts3d_mse, ga_loss arrays from a result dict.
    'inf' SNR entries are kept as np.inf for plotting on a finite x-axis.
    """
    snr, conf, mse, ga = [], [], [], []
    for r in result['sweep']:
        s = r['snr_db']
        snr.append(np.inf if s == 'inf' else float(s))
        conf.append(r['mean_conf'])
        mse.append(r['pts3d_mse'])
        ga.append(r['ga_loss'])
    return np.array(snr), np.array(conf), np.array(mse), np.array(
        [v if v is not None else np.nan for v in ga], dtype=float)


def label_for(result: dict, path: str) -> str:
    """Auto-generate a legend label from the result metadata."""
    channel = result.get('channel', 'awgn').upper()
    phase   = result.get('phase', 'A')
    stem    = os.path.splitext(os.path.basename(path))[0]
    # Remove redundant phase tag from filename stem (desktop_phaseB_awgn → desktop_awgn)
    stem_clean = stem.replace('_phaseC', '').replace('_phaseB', '').replace('_phaseA', '')
    return f"Phase {phase} {channel} ({stem_clean})"


def _detect_phases(results_list: list) -> set:
    """Return the set of phase labels ('A', 'B') found in results."""
    return {r.get('phase', 'A') for r in results_list}


def _make_title(phases: set) -> str:
    if phases == {'A'}:
        return 'DUSt3R + SemCom — Phase A: SNR Sweep Results'
    elif phases == {'B'}:
        return 'DUSt3R + SemCom — Phase B: SNR Sweep Results'
    elif phases == {'C'}:
        return 'DUSt3R + SemCom — Phase C: SNR Sweep Results'
    elif phases == {'B', 'C'}:
        return 'DUSt3R + SemCom — Phase B vs C Comparison: SNR Sweep'
    else:
        sorted_phases = ''.join(sorted(phases))
        return f'DUSt3R + SemCom — Phase {sorted_phases} Comparison: SNR Sweep'


def _make_filename(phases: set) -> str:
    if phases == {'A'}:
        return 'semcom_phaseA_results.png'
    elif phases == {'B'}:
        return 'semcom_phaseB_results.png'
    elif phases == {'C'}:
        return 'semcom_phaseC_results.png'
    elif phases == {'B', 'C'}:
        return 'semcom_phaseBC_comparison.png'
    else:
        sorted_phases = ''.join(sorted(phases))
        return f'semcom_phase{sorted_phases}_comparison.png'


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _finite_snr(snr: np.ndarray, placeholder: float = 25.0):
    """Replace np.inf SNR with a finite placeholder for the x-axis."""
    return np.where(np.isinf(snr), placeholder, snr)


def _snr_xticks(snr_vals, placeholder=25.0):
    ticks = sorted(set(_finite_snr(snr_vals, placeholder).tolist()))
    labels = ['∞' if abs(t - placeholder) < 0.01 else str(int(t)) for t in ticks]
    return ticks, labels


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_mean_conf(ax, results_list, paths, placeholder=25.0):
    ax.set_title('Mean Confidence vs SNR', fontsize=13, fontweight='bold')
    for result, path in zip(results_list, paths):
        snr, conf, _, _ = parse_sweep(result)
        xs = _finite_snr(snr, placeholder)
        baseline = result['baseline']['mean_conf']
        ax.axhline(baseline, linestyle='--', color='gray', linewidth=1,
                   label='_nolegend_')
        ax.plot(xs, conf, marker='o', linewidth=2, label=label_for(result, path))
    ticks, labels = _snr_xticks(snr, placeholder)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Mean Confidence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.annotate('― baseline (no channel)', xy=(0.98, 0.97),
                xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='gray')


def plot_conf_drop(ax, results_list, paths, placeholder=25.0):
    ax.set_title('Confidence Degradation (%) vs SNR', fontsize=13, fontweight='bold')
    for result, path in zip(results_list, paths):
        snr, conf, _, _ = parse_sweep(result)
        xs = _finite_snr(snr, placeholder)
        baseline = result['baseline']['mean_conf']
        drop_pct = (baseline - conf) / baseline * 100
        ax.plot(xs, drop_pct, marker='s', linewidth=2, label=label_for(result, path))
    ticks, labels = _snr_xticks(snr, placeholder)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Confidence Drop (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_pts3d_mse(ax, results_list, paths, placeholder=25.0):
    ax.set_title('Pointmap MSE vs SNR', fontsize=13, fontweight='bold')
    for result, path in zip(results_list, paths):
        snr, _, mse, _ = parse_sweep(result)
        xs = _finite_snr(snr, placeholder)
        # exclude the inf-SNR point (MSE=0, breaks log scale)
        finite_mask = np.isfinite(snr)
        ax.semilogy(xs[finite_mask], mse[finite_mask],
                    marker='^', linewidth=2, label=label_for(result, path))
    ticks, labels = _snr_xticks(snr, placeholder)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Pointmap MSE (log scale)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')


def plot_ga_loss(ax, results_list, paths, placeholder=25.0):
    ax.set_title('Global Alignment Loss vs SNR', fontsize=13, fontweight='bold')
    has_data = False
    for result, path in zip(results_list, paths):
        snr, _, _, ga = parse_sweep(result)
        xs = _finite_snr(snr, placeholder)
        valid = ~np.isnan(ga)
        failed = np.isnan(ga) & ~np.isinf(snr)  # NaN but finite SNR = GA failed
        if valid.any():
            has_data = True
            lbl = label_for(result, path)
            line, = ax.plot(xs[valid], ga[valid], marker='D', linewidth=2, label=lbl)
            # Mark failed GA points as red X on the x-axis
            if failed.any():
                ax.scatter(xs[failed], np.zeros(failed.sum()),
                           marker='x', s=120, linewidths=2.5,
                           color=line.get_color(), zorder=5,
                           label=f'{lbl} (GA FAILED)')
    if not has_data:
        ax.text(0.5, 0.5,
                'No GA data.\n(Need ≥3 images to enable\nPointCloudOptimizer)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray',
                bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.8))
    else:
        ticks, labels = _snr_xticks(snr, placeholder)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('GA Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.annotate('✕ = GA failed (degenerate point cloud)',
                    xy=(0.02, 0.03), xycoords='axes fraction',
                    fontsize=8, color='gray')


# ── Main ──────────────────────────────────────────────────────────────────────

def make_plots(result_paths: list, outdir: str | None = None):
    results_list = [load_result(p) for p in result_paths]
    phases = _detect_phases(results_list)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(_make_title(phases), fontsize=15, fontweight='bold', y=1.01)

    plot_mean_conf(axes[0, 0], results_list, result_paths)
    plot_conf_drop(axes[0, 1], results_list, result_paths)
    plot_pts3d_mse(axes[1, 0], results_list, result_paths)
    plot_ga_loss(axes[1, 1], results_list, result_paths)

    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, _make_filename(phases))
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'[Saved] {out_path}')
        # Also save individual plots
        for idx, (plot_fn, name) in enumerate([
            (plot_mean_conf, 'conf'),
            (plot_conf_drop, 'conf_drop'),
            (plot_pts3d_mse, 'mse'),
            (plot_ga_loss, 'ga_loss'),
        ]):
            fig_s, ax_s = plt.subplots(figsize=(7, 5))
            plot_fn(ax_s, results_list, result_paths)
            fig_s.tight_layout()
            p = os.path.join(outdir, f'semcom_{name}.png')
            fig_s.savefig(p, dpi=150, bbox_inches='tight')
            plt.close(fig_s)
            print(f'[Saved] {p}')
    else:
        plt.show()

    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot DUSt3R SemCom Phase A experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results', nargs='+',
                        help='JSON result file(s) from experiment_semcom.py')
    parser.add_argument('--outdir', default=None,
                        help='Directory to save plots (default: show interactively)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    make_plots(args.results, args.outdir)
