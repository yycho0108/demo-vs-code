#!/usr/bin/env python3

from typing import Optional
from natsort import natsorted
from pathlib import Path
from dataclasses import dataclass
import pickle
import cv2
from config import oc_cli
import numpy as np
import open3d as o3d


@dataclass
class Config:
    seq_path: str = '/tmp/sav10/'
    traj_path: str = '/tmp/sav2/hand.pkl'
    calib_path: str = '/tmp/sav10/cam.pkl'
    show_cloud: bool = False


def get_images(cfg: Config):
    colors = []
    depths = []

    for i, f in enumerate(natsorted(Path(cfg.seq_path).glob('color*.png'))):
        f_c = str(f)
        color = cv2.imread(f_c, cv2.IMREAD_COLOR)[..., ::-1]

        f_d = str(f).replace('color', 'depth')
        depth = cv2.imread(f_d, cv2.IMREAD_UNCHANGED) / 1000.0
        colors.append(color)
        depths.append(depth)
    colors = np.stack(colors, axis=0)
    depths = np.stack(depths, axis=0)
    return (colors, depths)


def np2o3d_img2pcd(
        color: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cb: Optional[np.ndarray] = None,
        normal: bool = True,
        depth_max: float = 3.0
):
    """
    Args:
        color: color image
        depth: depth image
        K: intrinsics
        T_cb: extrinsics ("camera_from_base" transform)
    """
    T_cb = o3d.core.Tensor.from_numpy(T_cb)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color.astype(np.uint8)),
            o3d.t.geometry.Image(depth.astype(np.float32))
        ),
        K,
        depth_scale=1.0,
        with_normals=normal,
        extrinsics=T_cb,
        depth_max=depth_max
    )
    return pcd.to_legacy()


@oc_cli
def main(cfg: Config):
    # load camera intrinsics & extrinsics.
    with open(cfg.calib_path, 'rb') as fp:
        data = pickle.load(fp)
        K = data['K']
        T = data['T']

    # load computed trajectory (output of client.py)
    with open(cfg.traj_path, 'rb') as fp:
        traj = pickle.load(fp)

    # optionally show the point cloud.
    if cfg.show_cloud:
        color, depth = get_images(cfg)
        clouds = [np2o3d_img2pcd(c, d, K,
                                 np.eye(4)) for (c, d) in
                  zip(color, depth)]

    # apply camera transform to `kpt`
    ps = []
    for c, k, r in zip(traj['cam'],
                       traj['kpt'],
                       traj['rgt']):
        c = np.asarray(c)
        k = np.asarray(k)
        t = c.reshape(-1, 1, 3)[-1]
        p = k.reshape(-1, 21, 3)[-1] + t
        ps.append(p)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # axis shows where the tag is (was) w.r.t. the camera
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    axis.transform(T)
    vis.add_geometry(axis)

    # Hand keypoints visualization
    kpts = o3d.geometry.PointCloud()
    kpts.points = o3d.utility.Vector3dVector(ps[0])
    kpts.colors = o3d.utility.Vector3dVector(np.zeros_like(ps[0]))
    vis.add_geometry(kpts)

    if cfg.show_cloud:
        cloud = o3d.geometry.PointCloud()
        vis.add_geometry(cloud)

    for j in range(100000):
        i = j % len(ps)

        # keypoints
        kpts.points = o3d.utility.Vector3dVector(ps[i])
        vis.update_geometry(kpts)

        # point cloud (optional)
        if cfg.show_cloud:
            cloud.points = clouds[i].points
            cloud.colors = clouds[i].colors
            vis.update_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    main()
