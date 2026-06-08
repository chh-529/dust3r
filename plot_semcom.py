#!/usr/bin/env python3
"""
SemCom Results Plotter
======================
Reads one or more JSON result files produced by eval_semcom.py (dataset-level
metrics) or experiment_semcom.py (qualitative SNR sweeps) and generates
publication-quality comparison plots.

It auto-detects which metrics are present and only plots those with data, so the
same script works for both the new dataset-level eval (regr3d_l2, chamfer,
abs_rel, delta125, ...) and the older experiment files (pts3d_mse, ga_loss).

Usage
-----
# Compare the three AWGN models:
  python plot_semcom.py \\
      results/eval_noisy_awgn.json \\
      results/eval_e2e_awgn_r0.125.json \\
      results/eval_e2e_awgn_r0.083.json \\
      --outdir figures/eval_compare_awgn
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ── Loading ─────────────────────────────────────────────────────────────────────

def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_series(result: dict, key: str):
    """
    Return (snr_array, value_array) for a metric ``key``.
    'inf' SNR is kept as np.inf; missing/None values become np.nan.
    """
    snr, val = [], []
    for r in result['sweep']:
        s = r['snr_db']
        snr.append(np.inf if s == 'inf' else float(s))
        v = r.get(key)
        val.append(np.nan if v is None else float(v))
    return np.array(snr, dtype=float), np.array(val, dtype=float)


def label_for(result: dict, path: str) -> str:
    """Auto-generate a legend label from the result metadata."""
    channel = result.get('channel', 'awgn').upper()
    if 'jscc_mode' in result:
        mode = 'JSCC' if result['jscc_mode'] == 'jscc' else 'noise-only'
    else:
        phase_map = {'A': 'noise-only', 'B': 'JSCC', 'C': 'JSCC (e2e)'}
        mode = phase_map.get(result.get('phase', 'A'), 'noise-only')
    stem = os.path.splitext(os.path.basename(path))[0]
    for tag in ('_phaseC', '_phaseB', '_phaseA'):
        stem = stem.replace(tag, '')

    # Clean SemCom ablation baseline (E2E fine-tuned, identity channel, no JSCC).
    if mode == 'JSCC' and 'identity' in stem:
        return f'{channel} SemCom baseline (no compression)'
    # E2E / JSCC dataset-eval files: express compression as 1/N from filename ratio.
    m = re.search(r'r([0-9.]+)', stem)
    if mode == 'JSCC' and m:
        ratio = float(m.group(1))
        n = round(1 / ratio) if ratio > 0 else 0
        return f'{channel} E2E (1/{n} compression)'
    if mode == 'noise-only':
        return f'{channel} noise-only (baseline)'
    return f'{channel} {mode} ({stem})'


def _scalar_metric(result: dict, key: str):
    """Single representative value of a metric (for a flat reference line)."""
    b = result.get('baseline') or {}
    if b.get(key) is not None:
        return float(b[key])
    for r in result['sweep']:
        v = r.get(key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
    return None


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _finite_snr(snr: np.ndarray, placeholder: float = 25.0):
    return np.where(np.isinf(snr), placeholder, snr)


def _snr_xticks(snr_vals, placeholder=25.0):
    ticks = sorted(set(_finite_snr(snr_vals, placeholder).tolist()))
    labels = ['∞' if abs(t - placeholder) < 0.01 else str(int(t)) for t in ticks]
    return ticks, labels


# Metric registry: key -> (title, ylabel, logscale, exclude_inf, group)
# Order here defines plotting order.  Arrows show the "good" direction.
#   group '3d'    : 3D reconstruction quality (vs GT pointcloud)
#   group 'depth' : view-1 depthmap quality
#   group 'diag'  : diagnostic — training objective & self-reported confidence
#                   (confounded across models; NO upper-bound line)
METRICS = [
    ('regr3d_l2', 'Pointmap L2 (scale-norm, conf-free)', 'Regr3D L2  ↓', False, False, '3d'),
    ('chamfer',   'Chamfer Distance',                     'Chamfer  ↓',    False, False, '3d'),
    ('acc',       'Accuracy (pred → GT)',                 'Acc  ↓',        False, False, '3d'),
    ('comp',      'Completeness (GT → pred)',             'Comp  ↓',       False, False, '3d'),
    ('pts3d_mse', 'Pointmap MSE',                         'MSE (log)  ↓',  True,  True,  '3d'),
    ('abs_rel',   'Depth Abs-Rel error',                  'AbsRel  ↓',     False, False, 'depth'),
    ('delta125',  'Depth  δ < 1.25',                      'δ<1.25  ↑',     False, False, 'depth'),
    ('task_loss', 'Task Loss (ConfLoss — diagnostic)',    'Task Loss',     False, False, 'diag'),
    ('mean_conf', 'Mean Confidence (self-reported)',      'Confidence  ↑', False, False, 'diag'),
    ('ga_loss',   'Global Alignment Loss',                'GA Loss  ↓',    False, True,  'diag'),
]

# group key -> (figure title, draw clean-DUSt3R upper-bound reference line?)
GROUPS = [
    ('3d',    '3D Reconstruction',                            True),
    ('depth', 'Depthmap (view-1)',                            True),
    ('diag',  'Diagnostic — training objective & confidence', False),
]

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _has_data(results_list, key, exclude_inf):
    for result in results_list:
        snr, val = get_series(result, key)
        mask = np.isfinite(val)
        if exclude_inf:
            mask &= np.isfinite(snr)
        if mask.any():
            return True
    return False


def plot_metric(ax, results_list, paths, key, title, ylabel,
                logscale=False, exclude_inf=False, placeholder=25.0,
                upper_bound=None):
    ax.set_title(title, fontsize=12, fontweight='bold')
    all_snr = None
    for i, (result, path) in enumerate(zip(results_list, paths)):
        snr, val = get_series(result, key)
        xs = _finite_snr(snr, placeholder)
        mask = np.isfinite(val)
        if exclude_inf:
            mask &= np.isfinite(snr)
        if not mask.any():
            continue
        all_snr = snr
        mk = MARKERS[i % len(MARKERS)]
        plot_fn = ax.semilogy if logscale else ax.plot
        plot_fn(xs[mask], val[mask], marker=mk, linewidth=2,
                label=label_for(result, path))
    # Clean DUSt3R upper-bound reference line (no channel, no compression).
    if upper_bound is not None:
        ub = _scalar_metric(upper_bound, key)
        if ub is not None and np.isfinite(ub):
            ax.axhline(ub, linestyle='--', color='black', linewidth=1.3,
                       alpha=0.7, label='Clean DUSt3R (no channel)')
    if all_snr is None:
        ax.text(0.5, 0.5, f'No {key} data', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return
    ticks, labels = _snr_xticks(all_snr, placeholder)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both' if logscale else 'major')


# ── Main ──────────────────────────────────────────────────────────────────────

def _channel_str(results_list):
    return '_'.join(sorted({r.get('channel', 'awgn') for r in results_list}))


def _make_group_fig(results_list, result_paths, metrics, group_title,
                    upper_bound, channel_label):
    """Build one figure for a group of metrics.  Returns the Figure."""
    ncol = min(len(metrics), 2)
    nrow = (len(metrics) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.8 * nrow),
                             squeeze=False)
    fig.suptitle(f'DUSt3R × SemCom — {group_title}  ({channel_label}, BlendedMVS val)',
                 fontsize=14, fontweight='bold', y=1.01)
    for idx, (key, title, ylabel, logscale, excl, _g) in enumerate(metrics):
        ax = axes[idx // ncol][idx % ncol]
        plot_metric(ax, results_list, result_paths, key, title, ylabel,
                    logscale, excl, upper_bound=upper_bound)
    for idx in range(len(metrics), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis('off')
    fig.tight_layout()
    return fig


def make_plots(result_paths, outdir=None, upper_bound_path=None):
    results_list = [load_result(p) for p in result_paths]
    upper_bound = load_result(upper_bound_path) if upper_bound_path else None
    channels = _channel_str(results_list)
    channel_label = ' vs '.join(sorted({r.get('channel', 'awgn').upper()
                                        for r in results_list}))

    any_plotted = False
    for gkey, gtitle, use_ub in GROUPS:
        metrics = [m for m in METRICS
                   if m[5] == gkey and _has_data(results_list, m[0], m[3] and m[4])]
        if not metrics:
            continue
        any_plotted = True
        # Upper-bound line only for quality groups, never for diagnostics.
        ub = upper_bound if use_ub else None
        fig = _make_group_fig(results_list, result_paths, metrics, gtitle,
                              ub, channel_label)

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            out_path = os.path.join(outdir, f'semcom_{channels}_{gkey}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f'[Saved] {out_path}')
            for key, title, ylabel, logscale, excl, _g in metrics:
                fig_s, ax_s = plt.subplots(figsize=(7, 5))
                plot_metric(ax_s, results_list, result_paths, key, title, ylabel,
                            logscale, excl, upper_bound=ub)
                fig_s.tight_layout()
                p = os.path.join(outdir, f'semcom_{key}.png')
                fig_s.savefig(p, dpi=150, bbox_inches='tight')
                plt.close(fig_s)
                print(f'[Saved] {p}')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    if not any_plotted:
        print('[plot_semcom] No plottable metrics found.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot DUSt3R SemCom experiment / eval results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results', nargs='+',
                        help='JSON result file(s) from eval_semcom.py or experiment_semcom.py')
    parser.add_argument('--outdir', default=None,
                        help='Directory to save plots (default: show interactively)')
    parser.add_argument('--upper_bound', default=None,
                        help='Result JSON (e.g. clean DUSt3R, no channel) drawn as a '
                             'horizontal dashed reference line on every metric panel.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    make_plots(args.results, args.outdir, args.upper_bound)
