#!/usr/bin/env python3

from dataclasses import dataclass
from config import oc_cli
from cam import MultiRSCamera, CameraConfig
from functools import partial
import time
import cv2
import open3d as o3d
import numpy as np
import requests
import json
import pickle
import threading

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


@dataclass
class Config:
    cam: CameraConfig = CameraConfig()
    device_id: str = '819312070397'

    fps: float = 30.0
    show: bool = False

    warmup: float = 1.0

    # host: str = '137.68.192.166'
    host: str = 'localhost'
    port: int = 8001
    port_in: int = 8002
    cam_path: str = '/tmp/cam.json'
    img_path: str = '/tmp/docker/img.png'
    out_path: str = '/tmp/docker/out.pkl'
    vid_mode: bool = False


@oc_cli
def main(cfg: Config):

    with open(cfg.cam_path, 'r') as fp:
        data = json.load(fp)
        data = {k: np.asarray(v, dtype=np.float32)
                for (k, v) in data.items()}
        K = data['K']
        T = data['T']  # T = cam_from_tag
        fx = K[0, 0]

        world_from_tag = np.eye(4)
        world_from_tag[:3, :3] = np.asarray([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        tag_from_cam = np.linalg.inv(T)
        world_from_cam = world_from_tag @ tag_from_cam

    cam_cfg = MultiRSCamera.Config.map_devices(cfg.cam,
                                               [cfg.device_id])
    predictor = ServerProxy(F'http://{cfg.host}:{cfg.port}')

    vis = None
    if cfg.show:
        vis = o3d.visualization.Visualizer()
        win = vis.create_window()

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        # axis.transform(T)
        vis.add_geometry(axis)

        kpts = o3d.geometry.PointCloud()
        zero = np.zeros((21, 3))
        kpts.points = o3d.utility.Vector3dVector(zero)
        kpts.colors = o3d.utility.Vector3dVector(zero)
        vis.add_geometry(kpts)

    with MultiRSCamera(cam_cfg).open() as cam:
        # warm-up
        n_warmup = int(max(1, cfg.warmup / 0.025))
        for _ in range(n_warmup):
            time.sleep(0.025)
            frame = cam()
        prev_stamp = frame['stamp']

        with SimpleXMLRPCServer((cfg.host, cfg.port_in),
                                requestHandler=RequestHandler) as server:
            state = {'prev_stamp': prev_stamp,
                     'ps_world': np.zeros((21, 3))
                     }

            def on_kpt(state):
                return state['ps_world'].tolist()
            server.register_introspection_functions()
            server.register_function(partial(on_kpt, state=state),
                                     'kpt')

            # loop...

            def step(state):
                frame = cam()

                # Skip old (or not sufficiently new) frames.
                stamp = frame['stamp']
                dt = stamp - state['prev_stamp']
                is_new = (np.greater(dt, 1000.0 / cfg.fps).all())
                if not is_new:
                    return
                state['prev_stamp'] = stamp

                # process frames.
                count: int = len(frame['stamp'])
                for j in range(count):
                    color_rgb = frame['color'][j]

                    # <- save img ->
                    if (not cfg.vid_mode):
                        cv2.imwrite(cfg.img_path, color_rgb[..., ::-1])
                    else:
                        # {vid}
                        writer = cv2.VideoWriter(
                            cfg.vid_path,
                            fourcc=cv2.VideoWriter_fourcc(
                                *"DIVX"),
                            fps=int(cfg.fps))
                        writer.write(color_rgb[..., ::-1])
                        writer.release()

                    # <- send img to HaMeR srv ->
                    if cfg.vid_mode:
                        # {vid}
                        with open(cfg.vid_path, 'rb') as fp:
                            resp = requests.post(
                                F'http://{cfg.host}:{cfg.port}',
                                files=dict(file=fp),
                                data=dict(focal=float(fx)))
                        traj = resp.json()
                    else:
                        # {img}
                        out = predictor.hand_img(cfg.img_path,
                                                 cfg.out_path,
                                                 float(fx))
                        with open(str(cfg.out_path), 'rb') as fp:
                            data = [pickle.load(fp)]
                        traj = []
                        for datum in data:
                            if datum is None:
                                continue
                            det = dict(kpt=datum['pred_keypoints_3d'],
                                       cam=datum['pred_cam_t_full'],
                                       rgt=datum['is_rights'])
                            traj.append(det)

                    # <- interpret `traj` from srv ->
                    ps = []
                    rs = []
                    for det in traj:
                        c, k, r = det['cam'], det['kpt'], det['rgt']
                        c = np.asarray(c)
                        k = np.asarray(k)
                        t = c.reshape(-1, 1, 3)[-1]
                        p = k.reshape(-1, 21, 3)[-1] + t
                        ps.append(p)
                        rs.append(r)

                    # <- update `ps_world` output ->
                    for i in range(len(ps)):
                        state['ps_world'] = (
                            ps[i] @ world_from_cam[: 3, : 3].T +
                            world_from_cam[: 3, 3]
                        )

                    if cfg.show and len(ps) > 0:
                        kpts.points = o3d.utility.Vector3dVector(
                            state
                            ['ps_world']
                        )

                        # app = o3d.visualization.gui.Application.instance
                        # app.post_to_main_thread(win, lambda: vis.update_geometry(kpts))

                        vis.update_geometry(kpts)
                        for _ in range(4):
                            vis.poll_events()
                            vis.update_renderer()

            def loop(state):
                while True:
                    step(state)
            # server.service_actions = partial(step, state=state)
            thread = threading.Thread(target=partial(loop, state=state),
                                      daemon=True)
            thread.start()
            server.serve_forever()
            # loop(state)


if __name__ == '__main__':
    main()
