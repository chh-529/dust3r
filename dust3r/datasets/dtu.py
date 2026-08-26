"""
DTU in DUSt3R's stereo-view format, for both fine-tuning and evaluation.

DTU ships images + calibrated cameras but no depth maps, so the ground truth here is
rendered from the official STL point clouds by datasets_preprocess/preprocess_dtu.py,
which must be run first. Coverage is object-and-table only; the rest is masked out
through valid_mask.

SPLITS
------
The standard MVSNet partition of DTU -- 79 train / 18 val / 22 test scans

    DTU(split='train', ROOT='data/dtu_processed')  # the 79 training scans
    DTU(split='val',   ROOT='data/dtu_processed')  # the 18 validation scans
    DTU(split='test',  ROOT='data/dtu_processed')  # the 22 evaluation scans
    DTU(split='none',    ROOT='data/dtu_processed')  # all 119 scans, no split
"""
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

# MVSNet's split of the 128 DTU scans. scans 78-81 are in no split and are absent from
# the official Rectified.zip; the remaining 9 unlisted scans have unusable ground truth.
MVSNET_SPLITS = dict(
    train=[2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44, 45, 46, 47,
           50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72, 74, 76,
           83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
           102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120, 121,
           122, 123, 124, 125, 126, 127, 128],
    val=[3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117],
    test=[1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110,
          114, 118],
)


# Helper functions
def scan_number(scan) -> int:
    """'scan114' -> 114"""
    return int(str(scan).replace('scan', ''))


class DTU(BaseStereoViewDataset):
    """DTU scans as image pairs, paired by the official pair.txt scores."""

    def __init__(self, *args, ROOT, split=None, scans=None, **kwargs):
        """
        Args:
            ROOT: data/dtu_processed
            split: 'train' / 'val' / 'test' or None for every scan present in ROOT.
            scans: optional further restriction, e.g. ['scan1', 'scan9']. Applied on top
                of `split`, so it can never pull in a scan the split excludes.
        """
        self.ROOT = ROOT
        super().__init__(*args, split=split, **kwargs)
        if split not in (None, *MVSNET_SPLITS):
            raise ValueError(f'{split = } is not one of {sorted(MVSNET_SPLITS)} or None')

        pairs = np.load(osp.join(ROOT, 'dtu_pairs.npy'))
        if split is not None:
            keep = set(MVSNET_SPLITS[split])
            pairs = pairs[[scan_number(s) in keep for s in pairs['scan']]]
            if len(pairs) == 0:
                raise ValueError(f"no {split} scans in {ROOT}")

        if scans is not None:
            pairs = pairs[np.isin(pairs['scan'], list(scans))]

        self.pairs = pairs
        self.scans = np.unique(self.pairs['scan'])

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scans)} scans'

    def _get_views(self, pair_idx, resolution, rng):
        scan, view_i, view_j, _score = self.pairs[pair_idx]
        scan_path = osp.join(self.ROOT, str(scan))

        views = []
        for view_index in (view_i, view_j):
            name = f'{int(view_index):08d}'
            image = imread_cv2(osp.join(scan_path, name + '.jpg'))
            meta = np.load(osp.join(scan_path, name + '.npz'))

            # rendered from the STL cloud, in millimetres; zero means "no ground truth"
            depthmap = meta['depthmap'].astype(np.float32)
            intrinsics = meta['intrinsics'].astype(np.float32)
            camera_pose = meta['cam2world'].astype(np.float32)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scan_path, name))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,   # cam2world
                camera_intrinsics=intrinsics,
                dataset='DTU',
                label=str(scan),
                instance=name))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = DTU(split='train', ROOT="data/dtu_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
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
        viz.show()
