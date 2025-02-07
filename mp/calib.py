#!/usr/bin/env python3

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

import cv2
from dt_apriltags import Detector

from config import oc_cli
from cam import MultiRSCamera, CameraConfig


@dataclass
class Config:
    # tag params
    tag_family: str = 'tag36h11'
    tag_size: float = 0.159
    tag_id: int = 0

    # cam params
    cam: CameraConfig = CameraConfig(
        img_width=424,
        img_height=240,
    )
    device_id: str = '233622074125'

    # app params
    cam_path: str = '/tmp/cam.json'


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def get_camera_pose(detector: Detector,
                    rgb: np.ndarray,
                    K: np.ndarray,
                    tag_size: float,
                    tag_id: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    Args:
    Return:
        out[0]: Homogeneous 4x4 matrix representing `cam_from_tag` transform.
        out[1]: Success flag
    """
    # Parse camera params.
    # fx, fy, cx, cy
    cam_param = K[[0, 1, 0, 1],
                  [0, 1, 2, 2]].tolist()

    # Detect tags.
    gray = rgb2gray(rgb)
    tags = detector.detect(gray.astype(np.uint8), True, cam_param, tag_size)
    tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}

    # Return if target was not found.
    if tag_id not in tags_dict:
        if len(tags_dict) > 0:
            print(F'tag_id = {tag_id} not in {tags_dict.keys()}')
        return None

    # FIXME(ycho): here, hardcoded assumption
    # that tag_id == 0
    R, t = tags_dict[tag_id]
    T = np.eye(4)
    T[..., :3, :3] = R.reshape(3, 3)
    T[..., :3, 3] = t.reshape(-1)
    return T


@oc_cli
def main(cfg: Config = Config()):
    cam_cfg = MultiRSCamera.Config.map_devices(cfg.cam,
                                               [cfg.device_id])
    detector = Detector(
        families=cfg.tag_family,
        quad_decimate=1
    )

    with MultiRSCamera(cam_cfg).open() as cam:
        frame = cam()
        prev_stamp = frame['stamp']

        while True:
            frame = cam()
            stamp = frame['stamp']

            # skip old frames.
            is_new = (np.greater(stamp, prev_stamp).all())
            if not is_new:
                continue

            # process new frame.
            prev_stamp = frame['stamp']
            color = frame['color'].squeeze(axis=0)
            T = get_camera_pose(detector, color,
                                cam.Ks.squeeze(axis=0),
                                cfg.tag_size,
                                cfg.tag_id)

            # save cam_from_tag transform to `out_file`.
            if T is not None:
                Path(cfg.cam_path).parent.mkdir(parents=True,
                                                exist_ok=True)
                with open(cfg.cam_path, 'w') as fp:
                    K = cam.Ks.squeeze(axis=0)
                    json.dump(dict(K=K.tolist(), T=T.tolist()), fp)
                return


if __name__ == '__main__':
    main()
