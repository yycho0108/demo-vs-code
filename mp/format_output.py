#!/usr/bin/env python3

from typing import Optional
from natsort import natsorted
from pathlib import Path
from dataclasses import dataclass
import pickle
import json
import cv2
from config import oc_cli
import numpy as np
import open3d as o3d


@dataclass
class Config:
    seq_path: str = '/tmp/sav003/'
    traj_path: str = '/tmp/sav003/traj.pkl'
    cam_path: str = '/tmp/sav003/cam.json'
    out_path: str = '/tmp/sav003/act.json'
    grasp_eps: float = 0.025
    show: bool = False
    show_cloud: bool = False


def pose_from_kpt(k: np.ndarray) -> np.ndarray:
    """
    Arg:
        k: A[..., 21, 3] MANO keypoint array.

    Return:
        A[..., 4,4] homogeneous matrix representing the tool pose
        (world_from_tool transform) formatted as [R,t; 0,1]
    """
    wrist = k[..., 0, :]
    index = k[..., 8, :]
    thumb = k[..., 4, :]
    u1 = index - wrist
    u2 = thumb - wrist

    z = np.cross(u1, u2)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)

    x = 0.5 * (u1 + u2)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)

    R = np.stack([x, y, z], axis=-1)
    T = [np.eye(4) for _ in range(len(k))]
    T = np.stack(T, axis=0)
    T[..., :3, :3] = R
    T[..., :3, 3] = 0.5 * (index + thumb)
    return T


def grasp_from_kpt(k: np.ndarray, eps: Optional[float] = 0.025) -> np.ndarray:
    """
    Arg:
        k: A[..., 21, 3] MANO keypoint array.
        eps: maximum distance between index-thumb tips for the hand to be "grasping".

    Return:
        A[...] boolean array representing the grasp flag.
    """
    wrist = k[..., 0, :]
    index = k[..., 8, :]
    thumb = k[..., 4, :]
    dist = index - thumb
    dist = np.linalg.norm(index - thumb,
                          axis=-1)
    return dist < eps


@oc_cli
def main(cfg: Config):
    # Load camera intrinsics & extrinsics.
    with open(cfg.cam_path, 'rb') as fp:
        data = pickle.load(fp)
        data = {k: np.asarray(v, dtype=np.float32)
                for (k, v) in data.items()}
        K = data['K']
        T = data['T']  # T = cam_from_tag

    # Load computed trajectory (output of client.py)
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

    kpts = np.stack(ps, axis=0)
    Ts = pose_from_kpt(kpts)
    Gs = grasp_from_kpt(kpts, cfg.grasp_eps)

    # Convert hand pose to world-frame.
    # Requires properly configuring `world_from_tag`.
    tag_from_cam = np.linalg.inv(T)
    world_from_tag = np.eye(4)
    world_from_tag[:3, :3] = np.asarray([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    world_from_tag[..., :3, 3] = [0.4, 0.0, 0.2 + 0.4]
    world_from_cam = world_from_tag @ tag_from_cam
    Ts = world_from_cam @ Ts
    with open(cfg.out_path, 'w') as fp:
        data = dict(tool_pose=Ts.tolist(),
                    grasp_flag=Gs.tolist())
        json.dump(data, fp)

    if cfg.show:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # axis shows where the tag is (was) w.r.t. the camera
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        axis.transform(world_from_tag)
        vis.add_geometry(axis)

        pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        vis.add_geometry(pose)

        for j in range(100000):
            i = j % len(ps)
            print(i)

            p = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            p.transform(Ts[i])
            pose.vertices = p.vertices
            vis.update_geometry(pose)

            for _ in range(4):
                vis.poll_events()
                vis.update_renderer()


if __name__ == '__main__':
    main()
