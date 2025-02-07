#!/usr/bin/env python3

import time
from typing import Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import cv2

from config import oc_cli
from cam import MultiRSCamera, CameraConfig


@dataclass
class Config:
    cam: CameraConfig = CameraConfig(
        img_width=424,
        img_height=240,
    )
    device_id: str = '233622074125'
    fps: float = 100.0
    show: bool = False
    save_dir: Optional[str] = '/tmp/sav'

    warmup: float = 1.0


@oc_cli
def main(cfg: Config):
    cam_cfg = MultiRSCamera.Config.map_devices(cfg.cam,
                                               [cfg.device_id])

    sav = None
    if cfg.save_dir is not None:
        sav = Path(cfg.save_dir)
        sav.mkdir(parents=True, exist_ok=True)
    with MultiRSCamera(cam_cfg).open() as cam:

        # Warm-up
        n_warmup = int(max(1, cfg.warmup / 0.025))
        for _ in range(n_warmup):
            time.sleep(0.025)
            frame = cam()
        prev_stamp = frame['stamp']

        # for t in tqdm(range(1000)):
        t = -1
        while True:
            frame = cam()

            # Skip old (or not sufficiently new) frames.
            stamp = frame['stamp']
            dt = stamp - prev_stamp
            is_new = (np.greater(dt, 1000.0 / cfg.fps).all())
            if not is_new:
                continue
            t = t + 1
            prev_stamp = stamp

            count: int = len(frame['stamp'])
            for j in range(count):
                if cfg.show:
                    cv2.imshow(F'color_{j:02d}', frame['color'][j, ..., ::-1])
                    cv2.imshow(F'depth_{j:02d}', frame['depth'][j])

                if (sav is not None):
                    depth_u16 = (
                        frame['depth'][j] *
                        cam_cfg.cams[j].depth_scale).astype(
                        np.uint16)
                    cv2.imwrite(F'{sav}/color_{j:02d}_{t:03d}.png',
                                frame['color'][j, ..., ::-1])
                    cv2.imwrite(F'{sav}/depth_{j:02d}_{t:03d}.png', depth_u16)
                    with open(F'{sav}/stamp_{j:02d}_{t:03d}.txt', 'w') as fp:
                        fp.write(F'{stamp[j]}')

            if cfg.show:
                k = cv2.waitKey(1)
                if (k & 0xFF) == ord('q'):
                    break

            pass


if __name__ == '__main__':
    main()
