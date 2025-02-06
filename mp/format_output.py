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


def pose_from_kpt(k):
    wrist = k[..., 0, :]
    index = k[..., 8, :]
    thumb = k[..., 4, :]
    u1 = index - wrist
    u2 = thumb - wrist

    z = np.cross(u1, u2)
    z /= np.linalg.norm(z)
    x = 0.5 * (u1 + u2)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.stack([x, y, z], axis=-1)
    T = [np.eye(4) for _ in range(len(k))]
    T = np.stack(T, axis=0)
    T[..., :3, :3] = R
    T[..., :3, 3] = 0.5 * (index + thumb)
    return T


@oc_cli
def main(cfg: Config):
    # load camera intrinsics & extrinsics.
    with open(cfg.calib_path, 'rb') as fp:
        data = pickle.load(fp)
        K = data['K']
        T = data['T']  # T = cam_from_tag

    # load computed trajectory (output of client.py)
    with open(cfg.traj_path, 'rb') as fp:
        traj = pickle.load(fp)

    # Compute camera-frame tool pose from keypoints.
    ps = []
    for c, k, r in zip(traj['cam'],
                       traj['kpt'],
                       traj['rgt']):
        c = np.asarray(c)
        k = np.asarray(k)
        t = c.reshape(-1, 1, 3)[-1]
        p = k.reshape(-1, 21, 3)[-1] + t
        ps.append(p)
    Ts = pose_from_kpt(np.stack(ps, axis=0))

    # (optional) Convert to tag-frame
    Ts = np.linalg.inv(T)[None] @ Ts

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # axis shows where the tag is (was) w.r.t. the camera
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    axis.transform(T)
    vis.add_geometry(axis)

    pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    vis.add_geometry(pose)
    Ts = pose_from_kpt(np.stack(ps, axis=0))

    for j in range(100000):
        i = j % len(ps)

        p = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        p.transform(Ts[i])
        pose.vertices = p.vertices
        vis.update_geometry(pose)

        for _ in range(128):
            vis.poll_events()
            vis.update_renderer()


if __name__ == '__main__':
    main()
