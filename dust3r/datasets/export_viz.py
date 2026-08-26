"""
Export DUSt3R stereo-view pairs to .glb scene files for offline visual inspection.

Runs on headless servers: unlike dust3r.viz.SceneViz.show() (needs GLU + a display),
trimesh.Scene.export() only writes geometry to disk, so it works with no GUI at all.
Open the resulting .glb with any local glTF viewer, e.g. https://gltf-viewer.donmccurdy.com/.

--dataset takes the same eval()'d spec string as semcom.train's --train_dataset /
--val_dataset (see semcom/train.py), so any class importable from dust3r.datasets works:

    python dust3r/datasets/export_viz.py \
        --dataset "DTU(split='test', ROOT='data/dtu_processed', resolution=224)" \
        --n 5 --out_dir dust3r/datasets/viz_out/dtu

    python dust3r/datasets/export_viz.py \
        --dataset "Co3d(split='train', ROOT='data/co3d_subset_processed', resolution=224)" \
        --n 5 --out_dir dust3r/datasets/viz_out/co3d
"""
import argparse
import os

import numpy as np

from dust3r.datasets import *  # noqa: F401,F403 -- brings dataset classes into eval() scope
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    help="eval()'d dataset spec, e.g. "
                         "\"DTU(split='test', ROOT='data/dtu_processed', resolution=224)\"")
    ap.add_argument('--n', type=int, default=5, help='number of random pairs to export')
    ap.add_argument('--out_dir', default='dust3r/datasets/viz_out')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    dataset = eval(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(dataset))[:args.n]

    for i, idx in enumerate(indices):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))

        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)

        out_path = os.path.join(args.out_dir, f'{i:03d}_idx{idx}.glb')
        viz.scene.export(out_path)
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
