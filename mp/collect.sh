#!/usr/bin/env bash

# == CONFIGURE ==
DEVICE_ID='819312070397' # replace this after running `python3 cam.py`
SAVE_DIR='/tmp/sav005'
FPS=5
TAG_ID=0 # you may need to replace this with another ID

VID_FILE="out.mp4"   # temporary filename for video (output of convert_to_video.py)
CAM_FILE="cam.json"  # camera intrinsics / extrinsics (output of calib.py)
TRAJ_FILE="traj.pkl" # hand trajectory (output from client.py)
ACT_FILE="act.json"  # final output (output of format_output.py)

# (0) figure out the camera serial number
# python3 cam.py

# (1) calibrate and dump camera intrinsics & extrinsics parameters.
python3 calib.py device_id="${DEVICE_ID}" cam_path="${SAVE_DIR}/${CAM_FILE}" tag_id="${TAG_ID}"

# (2) record demonstration data.
python3 record.py save_dir="${SAVE_DIR}" cam_path="${SAVE_DIR}/${CAM_FILE}" device_id="${DEVICE_ID}" fps="${FPS}" show=1 

# (3) convert demonstration data to video format (to send to server)
python3 convert_to_video.py seq_path="${SAVE_DIR}" vid_path="${VID_FILE}"

# (4) send demonstration data to server.
python3 client.py vid_path="${SAVE_DIR}/${VID_FILE}" cam_path="${SAVE_DIR}/${CAM_FILE}" out_path="${SAVE_DIR}/${TRAJ_FILE}"

# (4-1) visualize hand trajectory.
# python3 visualize.py seq_path="${SAVE_DIR}" traj_path="${SAVE_DIR}/${TRAJ_FILE}" cam_path="${SAVE_DIR}/${CAM_FILE}" #show_cloud=1

# (5) load hand trajectory convert to robot actions. 
python3 format_output.py seq_path="${SAVE_DIR}" traj_path="${SAVE_DIR}/${TRAJ_FILE}" cam_path="${SAVE_DIR}/${CAM_FILE}" out_path="${SAVE_DIR}/${ACT_FILE}"
