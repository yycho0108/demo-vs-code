#!/usr/bin/env python3

import contextlib
import os
from pathlib import Path
from dataclasses import dataclass
from config import oc_cli
import cv2
from natsort import natsorted
import subprocess


@dataclass
class Config:
    seq_path: str = ''
    vid_path: str = ''
    txt_name: str = 'input.txt'


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@oc_cli
def main(cfg: Config):
    last_stamp = None

    with working_directory(cfg.seq_path):
        with open(F'{cfg.txt_name}', 'w') as fp:
            for i, txt in enumerate(
                    natsorted(Path(cfg.seq_path).glob('stamp*.txt'))):
                with open(str(txt), 'r') as fp_t:
                    stamp = float(fp_t.read())
                if last_stamp is not None and stamp <= last_stamp:
                    continue

                color = txt.with_suffix('.png')
                color = color.parent / color.name.replace('stamp', 'color')

                if i > 0:
                    duration = max(0.04, stamp - last_stamp)
                    duration = int(duration)  # ??
                    fp.write(F"duration {duration}\n")
                fp.write(F"file '{color.name}'\n")
                last_stamp = stamp
        # write video using ffmpeg.
        subprocess.run(
            F'/usr/bin/ffmpeg -f concat -i {cfg.seq_path}/{cfg.txt_name} -vcodec libx264 -crf 20 -pix_fmt yuv420p -vf "settb=1/1000,setpts=PTS/1000" -vsync vfr -r 1000 {cfg.vid_path}',
            shell=True)


if __name__ == '__main__':
    main()
