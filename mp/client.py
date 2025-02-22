#!/usr/bin/env python3

from dataclasses import dataclass
import pickle
import json
from pathlib import Path
import requests
import cv2
from config import oc_cli
from cam import MultiRSCamera
import numpy as np


@dataclass
class Config:
    host: str = '137.68.192.166'
    port: int = 5000
    vid_path: str = '/tmp/sav2/out2.mp4'
    out_path: str = '/tmp/sav2/hand.pkl'
    cam_path: str = '/tmp/cam.json'


@oc_cli
def main(cfg: Config):
    with open(cfg.cam_path, 'r') as fp:
        K = np.asarray(json.load(fp)['K'], dtype=np.float32)
        fx = K[0, 0]

    with open(cfg.vid_path, 'rb') as fp:
        resp = requests.post(F'http://{cfg.host}:{cfg.port}',
                             files=dict(file=fp),
                             data=dict(focal=float(fx)))
        out = resp.json()
        with open(cfg.out_path, 'wb') as fp:
            pickle.dump(out, fp)


if __name__ == '__main__':
    main()
