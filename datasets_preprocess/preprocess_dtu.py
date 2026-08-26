"""
Turn DTU into DUSt3R's format: render per-pixel depth from the official STL point cloud
(DTU ships images + calibrated cameras but no depth maps), pairing views via pair.txt.
Coverage is object-only, matching the DTU protocol; the rest is masked out via
valid_mask.

Run once per DTU subdir, pointing both at the same --output_dir (scan numbers never
collide between dtu-test and dtu-train, and pairs auto-merge instead of overwriting):

    python datasets_preprocess/preprocess_dtu.py --dtu_dir <dir> \
        --subdir dtu-test --output_dir data/dtu_processed --srcs_per_ref 3
    python datasets_preprocess/preprocess_dtu.py --dtu_dir <dir> \
        --subdir dtu-train --output_dir data/dtu_processed --srcs_per_ref 3

Keep --pairs_per_scan at its default (0, no cap). pair.txt's score rewards short
baselines, so capping it clusters every kept view on one side of the object -- breaks
anything needing the full camera cap, e.g. semcom.dtu_lr's left/right split.

Output, per scan:
    scanN/<view>.jpg   image, resized to --render_size
    scanN/<view>.npz   depthmap (float16, mm), intrinsics, cam2world
    dtu_pairs.npy      (scan, view_i, view_j, score)
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm


def read_cam(path):
    """MVSNet cam file -> (world2cam 4x4, intrinsics 3x3)."""
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    ext = np.array([list(map(float, lines[i].split())) for i in range(1, 5)], np.float64)
    itr = np.array([list(map(float, lines[i].split())) for i in range(7, 10)], np.float64)
    return ext, itr


def render_depth(points, world2cam, K, H, W, chunk=4_000_000):
    """
    Z-buffer the point cloud into one camera.

    Args:
        points: (N, 3) float32 torch tensor, world coordinates
        world2cam: (4, 4), K: (3, 3), H, W: output size
    Returns:
        (H, W) float32 depth in the same unit as `points`; 0 where nothing projected.
    """
    buf = torch.full((H * W,), float('inf'), dtype=torch.float32)
    R = torch.from_numpy(world2cam[:3, :3]).float()
    t = torch.from_numpy(world2cam[:3, 3]).float()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for i in range(0, points.shape[0], chunk):
        p = points[i:i + chunk]
        pc = p @ R.T + t # world -> camera
        z = pc[:, 2]
        ok = z > 1e-6
        if not bool(ok.any()):
            continue
        pc, z = pc[ok], z[ok]
        u = (fx * pc[:, 0] / z + cx).round().long()
        v = (fy * pc[:, 1] / z + cy).round().long()
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not bool(inside.any()):
            continue
        # amin keeps the nearest surface, i.e. hides points occluded by the object
        buf.scatter_reduce_(0, (v[inside] * W + u[inside]), z[inside],
                            reduce='amin', include_self=True)

    depth = buf.reshape(H, W)
    depth[torch.isinf(depth)] = 0.0
    return depth.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtu_dir', default='/tmp2/b12902145/dataset/dtu')
    ap.add_argument('--subdir', default='dtu-test',
                    help='scan directory under --dtu_dir, in MVSNet layout')
    ap.add_argument('--output_dir', default='data/dtu_processed')
    ap.add_argument('--render_size', type=int, nargs=2, default=[800, 600],
                    help='(W, H) to render and store at; DUSt3R resizes to 512x384 later')
    ap.add_argument('--pairs_per_scan', type=int, default=0,
                    help='cap on pairs per scan, by descending score; 0 means no cap '
                         '(required by downstream code that needs the full camera cap, '
                         'e.g. semcom.dtu_lr)')
    ap.add_argument('--srcs_per_ref', type=int, default=1,
                    help='neighbours kept per reference view, best-scoring first')
    ap.add_argument('--max_points', type=int, default=12_000_000,
                    help='subsample the STL cloud above this, for speed')
    args = ap.parse_args()

    W, H = args.render_size
    scan_root = osp.join(args.dtu_dir, args.subdir)
    scans = sorted((s for s in os.listdir(scan_root) if s.startswith('scan')),
                   key=lambda s: int(s.replace('scan', '')))
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = []

    for scan in tqdm(scans, desc='scans'):
        sdir = osp.join(scan_root, scan)
        odir = osp.join(args.output_dir, scan)
        os.makedirs(odir, exist_ok=True)

        n = int(scan.replace('scan', ''))
        ply = osp.join(args.dtu_dir, 'Points', 'stl', f'stl{n:03d}_total.ply')
        if not osp.isfile(ply):
            print(f'  ! {scan}: no {osp.basename(ply)}, skipped')
            continue

        cloud = trimesh.load(ply, process=False)
        pts = np.asarray(cloud.vertices, dtype=np.float32)
        if pts.shape[0] > args.max_points:
            sel = np.random.RandomState(0).choice(pts.shape[0], args.max_points, False)
            pts = pts[sel]
        pts_t = torch.from_numpy(pts)

        # which views this scan needs: the top pairs from the official pair.txt
        with open(osp.join(sdir, 'pair.txt')) as f:
            lines = [l.strip() for l in f.readlines()]
        n_ref = int(lines[0])
        scan_pairs = []
        for i in range(n_ref):
            ref = int(lines[1 + 2 * i])
            toks = lines[2 + 2 * i].split()
            if len(toks) < 3:
                continue
            # pair.txt lists neighbours as (view, score) pairs, best-scoring first
            for k in range(args.srcs_per_ref):
                if 2 * k + 2 >= len(toks):
                    break
                scan_pairs.append((ref, int(toks[2 * k + 1]), float(toks[2 * k + 2])))
        scan_pairs.sort(key=lambda x: -x[2])
        if args.pairs_per_scan > 0:
            scan_pairs = scan_pairs[:args.pairs_per_scan]
        needed = sorted({v for p in scan_pairs for v in p[:2]})

        for view in needed:
            # dtu-test ships jpg, download_dtu_rectified.py writes the original png
            img_path = next((p for ext in ('.jpg', '.png')
                             if osp.isfile(p := osp.join(sdir, 'images',
                                                         f'{view:08d}{ext}'))), None)
            cam_path = osp.join(sdir, 'cams', f'{view:08d}_cam.txt')
            if img_path is None or not osp.isfile(cam_path):
                continue

            img = cv2.imread(img_path)
            h0, w0 = img.shape[:2]
            world2cam, K = read_cam(cam_path)

            # scale intrinsics with the image so depth, K and pixels stay consistent
            K = K.copy()
            K[0] *= W / w0
            K[1] *= H / h0
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            depth = render_depth(pts_t, world2cam, K, H, W)

            cam2world = np.linalg.inv(world2cam)
            cv2.imwrite(osp.join(odir, f'{view:08d}.jpg'), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savez_compressed(osp.join(odir, f'{view:08d}.npz'),
                                depthmap=depth.astype(np.float16),
                                intrinsics=K.astype(np.float32),
                                cam2world=cam2world.astype(np.float32))

        cov = 0.0
        for ref, src, score in scan_pairs:
            pairs.append((scan, ref, src, score))
        if needed:
            d = np.load(osp.join(odir, f'{needed[0]:08d}.npz'))['depthmap']
            cov = float((d > 0).mean())
        print(f'  {scan}: {len(pts):,} pts, {len(needed)} views, '
              f'{len(scan_pairs)} pairs, depth coverage {cov:.1%}')

    arr = np.array(pairs, dtype=[('scan', 'U10'), ('view_i', 'i4'),
                                 ('view_j', 'i4'), ('score', 'f4')])

    pairs_path = osp.join(args.output_dir, 'dtu_pairs.npy')
    if osp.isfile(pairs_path):
        existing = np.load(pairs_path)
        overlap = set(existing['scan']) & set(arr['scan'])
        if overlap:
            raise RuntimeError(
                f'{pairs_path} already has pairs for {sorted(overlap)}; delete it (or '
                'the whole --output_dir) before reprocessing the same scans')
        arr = np.concatenate([existing, arr])
    np.save(pairs_path, arr)
    print(f'\n{len(arr)} pairs from {len(set(arr["scan"]))} scans -> {pairs_path}')

    # stale once new scans are merged in; semcom.dtu_lr regenerates it on next use
    azimuth_cache = osp.join(args.output_dir, 'dtu_view_azimuth.npy')
    if osp.isfile(azimuth_cache):
        os.remove(azimuth_cache)
        print(f'removed stale {azimuth_cache}')


if __name__ == '__main__':
    main()
