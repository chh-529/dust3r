#!/usr/bin/env python3
"""
DUSt3R + SemCom Interactive Demo
=================================
Gradio web UI that lets you upload images and choose:
  - Channel type (None / AWGN / Rayleigh)
  - SNR slider
  - Phase (A: identity JSCC  /  B: trained linear JSCC)

The model reconstructs the 3D scene and shows how the wireless channel
degrades reconstruction quality in real time.

Usage
-----
# Phase A only (no JSCC checkpoint needed)
CUDA_VISIBLE_DEVICES=1 python demo_semcom.py \\
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# Phase B (load trained JSCC checkpoint)
CUDA_VISIBLE_DEVICES=1 python demo_semcom.py \\
    --weights     checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --jscc_weights checkpoints/jscc_phaseB_awgn_k512.pth
"""

import argparse
import copy
import functools
import math
import os
import tempfile

import gradio
import torch
import numpy as np
import matplotlib.pyplot as pl

from dust3r.demo import (
    _convert_scene_output_to_glb,
    get_3D_model_from_scene,
    set_scenegraph_options,
    set_print_with_timestamp,
)
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model_semcom import load_semcom_model, load_semcom_model_phaseB

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True


# ── Model loader ──────────────────────────────────────────────────────────────

def build_model(weights: str, jscc_weights: str | None, device: str,
                channel: str, snr_db: float):
    """
    Build the appropriate model variant based on UI settings.

    channel == 'none'  → clean DUSt3R (no SemCom)
    channel != 'none'  → SemCom wrapper
      jscc_weights provided → Phase B (trained Linear JSCC)
      jscc_weights is None  → Phase A (identity JSCC)
    """
    if channel == 'none':
        # Clean baseline: load Phase A with SNR=inf (identity path)
        return load_semcom_model(weights, device, snr_db=float('inf'), verbose=False)

    if jscc_weights is not None:
        # Phase B: load trained JSCC encoder/decoder
        return load_semcom_model_phaseB(
            weights, jscc_weights, device, snr_db=snr_db, verbose=False)
    else:
        # Phase A: identity JSCC, direct noise injection
        return load_semcom_model(
            weights, device, snr_db=snr_db, channel=channel, verbose=False)


# ── Reconstruction function ───────────────────────────────────────────────────

def get_reconstructed_scene(
    outdir, weights, jscc_weights, device, image_size, silent,
    filelist, channel, snr_db,
    schedule, niter, min_conf_thr,
    as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    scenegraph_type, winsize, refid,
):
    if not filelist:
        return None, None, None

    # Build model with current channel / SNR settings
    model = build_model(weights, jscc_weights, device, channel, float(snr_db))

    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    sg = scenegraph_type
    if sg == 'swin':
        sg = f'swin-{int(winsize)}'
    elif sg == 'oneref':
        sg = f'oneref-{int(refid)}'

    pairs = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = (GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        try:
            scene.compute_global_alignment(
                init='mst', niter=int(niter), schedule=schedule, lr=0.01)
        except Exception as e:
            print(f'[GA failed: {e}]')

    outfile = get_3D_model_from_scene(
        outdir, silent, scene, min_conf_thr, as_pointcloud,
        mask_sky, clean_depth, transparent_cams, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max(d.max() for d in depths)
    depths = [d / depths_max for d in depths]
    confs_max = max(d.max() for d in confs)
    confs = [cmap(d / confs_max) for d in confs]

    gallery = []
    for i in range(len(rgbimg)):
        gallery.append(rgbimg[i])
        gallery.append(rgb(depths[i]))
        gallery.append(rgb(confs[i]))

    del model
    torch.cuda.empty_cache()

    return scene, outfile, gallery


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def main_demo(tmpdirname, weights, jscc_weights, device, image_size, server_name,
              server_port, silent=False):
    recon_fn = functools.partial(
        get_reconstructed_scene,
        tmpdirname, weights, jscc_weights, device, image_size, silent,
    )

    model_from_scene_fn = functools.partial(get_3D_model_from_scene, tmpdirname, silent)

    phase_label = 'Phase B (trained JSCC)' if jscc_weights else 'Phase A (identity JSCC)'

    with gradio.Blocks(title='DUSt3R × SemCom Demo') as demo:
        scene_state = gradio.State(None)

        gradio.HTML(
            '<h2 style="text-align:center;">DUSt3R × SemCom Demo</h2>'
            f'<p style="text-align:center; color:#888;">{phase_label}</p>'
        )

        with gradio.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gradio.Column(scale=1):
                inputfiles = gradio.File(file_count='multiple', label='Input images')

                gradio.HTML('<b>📡 Channel settings</b>')
                channel = gradio.Radio(
                    choices=['none', 'awgn', 'rayleigh'],
                    value='none',
                    label='Channel type',
                    info='"none" = clean DUSt3R baseline',
                )
                snr_db = gradio.Slider(
                    label='SNR (dB)',
                    value=10.0, minimum=0.0, maximum=30.0, step=1.0,
                    info='Lower = more noise. Only used when channel ≠ none.',
                )

                gradio.HTML('<b>⚙️ Reconstruction settings</b>')
                with gradio.Row():
                    schedule = gradio.Dropdown(
                        ['linear', 'cosine'], value='linear', label='GA schedule')
                    niter = gradio.Number(
                        value=300, precision=0, minimum=0, maximum=5000,
                        label='GA iterations')
                with gradio.Row():
                    scenegraph_type = gradio.Dropdown(
                        [('complete: all pairs', 'complete'),
                         ('swin: sliding window', 'swin'),
                         ('oneref: one vs all', 'oneref')],
                        value='complete', label='Scene graph', interactive=True)
                    winsize = gradio.Slider(
                        label='Window size', value=1, minimum=1, maximum=2,
                        step=1, visible=False)
                    refid = gradio.Slider(
                        label='Ref image id', value=0, minimum=0, maximum=1,
                        step=1, visible=False)

                run_btn = gradio.Button('▶  Run reconstruction', variant='primary')

            # ── Right column: outputs ────────────────────────────────────────
            with gradio.Column(scale=2):
                outmodel = gradio.Model3D(label='3D scene')
                with gradio.Row():
                    min_conf_thr = gradio.Slider(
                        label='min_conf_thr', value=3.0, minimum=1.0,
                        maximum=20, step=0.1)
                    cam_size = gradio.Slider(
                        label='cam_size', value=0.05, minimum=0.001,
                        maximum=0.1, step=0.001)
                with gradio.Row():
                    as_pointcloud    = gradio.Checkbox(value=False, label='Pointcloud')
                    mask_sky         = gradio.Checkbox(value=False, label='Mask sky')
                    clean_depth      = gradio.Checkbox(value=True,  label='Clean depth')
                    transparent_cams = gradio.Checkbox(value=False, label='Transparent cams')
                outgallery = gradio.Gallery(
                    label='RGB | Depth | Confidence', columns=3, height='auto')

        # ── Common input list ────────────────────────────────────────────────
        recon_inputs = [
            inputfiles, channel, snr_db,
            schedule, niter, min_conf_thr,
            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
            scenegraph_type, winsize, refid,
        ]
        viz_inputs = [
            scene_state, min_conf_thr, as_pointcloud, mask_sky,
            clean_depth, transparent_cams, cam_size,
        ]

        # ── Events ──────────────────────────────────────────────────────────
        scenegraph_type.change(
            set_scenegraph_options,
            inputs=[inputfiles, winsize, refid, scenegraph_type],
            outputs=[winsize, refid])
        inputfiles.change(
            set_scenegraph_options,
            inputs=[inputfiles, winsize, refid, scenegraph_type],
            outputs=[winsize, refid])

        run_btn.click(
            fn=recon_fn,
            inputs=recon_inputs,
            outputs=[scene_state, outmodel, outgallery])

        for component in (min_conf_thr, cam_size):
            component.release(
                fn=lambda *a: model_from_scene_fn(*a),
                inputs=viz_inputs, outputs=outmodel)

        for component in (as_pointcloud, mask_sky, clean_depth, transparent_cams):
            component.change(
                fn=lambda *a: model_from_scene_fn(*a),
                inputs=viz_inputs, outputs=outmodel)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='DUSt3R + SemCom Gradio demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', required=True,
                   help='DUSt3R checkpoint path.')
    p.add_argument('--jscc_weights', default=None,
                   help='(Phase B) Trained JSCC checkpoint from train_semcom_phaseB.py. '
                        'Omit for Phase A (identity JSCC).')
    p.add_argument('--device', default='cuda')
    p.add_argument('--image_size', type=int, default=512, choices=[224, 512])
    p.add_argument('--server_port', type=int, default=7861,
                   help='Gradio server port.')
    p.add_argument('--local_network', action='store_true',
                   help='Bind to 0.0.0.0 (accessible from local network).')
    p.add_argument('--silent', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_print_with_timestamp()
    server_name = '0.0.0.0' if args.local_network else '127.0.0.1'
    with tempfile.TemporaryDirectory(suffix='_semcom_demo') as tmpdirname:
        main_demo(
            tmpdirname=tmpdirname,
            weights=args.weights,
            jscc_weights=args.jscc_weights,
            device=args.device,
            image_size=args.image_size,
            server_name=server_name,
            server_port=args.server_port,
            silent=args.silent,
        )
