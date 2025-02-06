#!/usr/bin/env python3

from dataclasses import dataclass
import pickle
from pathlib import Path
from xmlrpc.client import ServerProxy

import cv2

from config import oc_cli
from cam import MultiRSCamera


@dataclass
class Config:
    host: str = '137.68.192.166'
    port: int = 8001
    vid_path: str = ''
    out_path: str = ''


@oc_cli
def main(cfg: Config):
    from xmlrpc.client import ServerProxy
    srv = ServerProxy(F'http://{cfg.host}:{cfg.port}')

    if not Path(cfg.vid_path).is_file():
        raise FileNotFoundError(F'Video = {cfg.vid_path} does not exist')

    Path(cfg.out_path).parent.mkdir(parents=True,
                                    exist_ok=True)
    out = srv.hand(cfg.vid_path,
                   cfg.out_path,
                   616.0)


if __name__ == '__main__':
    main()
