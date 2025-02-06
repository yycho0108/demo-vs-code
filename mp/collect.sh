#!/usr/bin/env bash

DEVICE_ID='819312070397'
SAVE_DIR='/tmp/sav002'

# NOTE(ycho): basename only
VID_FILE="out.mp4" 
CAM_FILE="cam.pkl"
TRAJ_FILE="traj.pkl"

# (0) figure out the camera serial number
# python3 cam.py

# (1) calibrate and dump camera intrinsics & extrinsics parameters.
python3 calib.py device_id="${DEVICE_ID}" out_file="${SAVE_DIR}/${CAM_FILE}"

# (2) record demonstration data.
python3 record.py save_dir="${SAVE_DIR}" calib_file="${SAVE_DIR}/${CAM_FILE}" show=1 device_id="${DEVICE_ID}" fps=5

# (3) convert demonstration data to video format (to send to server)
python3 convert_to_video.py seq_path="${SAVE_DIR}" vid_path="${VID_FILE}"

# (4) send demonstration data to server.
python3 client.py vid_path="${SAVE_DIR}/${VID_FILE}" calib_path="${SAVE_DIR}/${CAM_FILE}" out_path="${SAVE_DIR}/${TRAJ_FILE}"

# (4-1) visualize hand trajectory.
# python3 visualize.py seq_path="${SAVE_DIR}" traj_path="${SAVE_DIR}/${TRAJ_FILE}" calib_path="${SAVE_DIR}/${CAM_FILE}" #show_cloud=1

# (5) load hand trajectory convert to robot actions. 
