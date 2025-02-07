import sys
sys.path.append('..')
import numpy as np
import pickle
from env.bimanual_bullet import PickCubeEnv
from scipy.spatial.transform import Rotation as R
from mp.iface import Command
import pybullet as p
from tqdm.auto import tqdm

from mp.ompl_wrapper import MotionPlanner


def main():
    cfg = {
        "seed": 42,
        "gui": True,
        "sim_hz": 240,
        "control_hz": 20,
        "q_init": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "render": {
            "width": 640,
            "height": 480,
            "cam": {
                "pos": [1, 1, 1],
                "lookat": [0.3, 0, 0.6],
                "up": [0, 0, 1],
                "fov": 60,
            }
        },
        "ctrl": {
            "gripper_gain": 0.03,
        },
    }

    env = PickCubeEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    joint_positions = env.get_joint_positions()
    left_tool_pose, right_tool_pose = env.get_tool_poses()
    object_pose_dict = env.get_object_poses()
    cube_pose = object_pose_dict["cube"]

    # Solve IK for pre-defined poses
    up_left_tool_pose = np.concatenate([cube_pose[:3], left_tool_pose[3:]])
    up_left_tool_pose[2] += 0.1
    z_rot_candidates = np.linspace(-np.pi/2, 0, 10)
    for z_rot in z_rot_candidates:
        up_left_tool_pose[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(up_left_tool_pose)
        up_q_left = env.solve_tool_ik(up_left_tool_pose, "left", check_collision=True)
        if up_q_left is not None:
            break
    else:
        raise ValueError("No valid up pose found")

    # Do motion planning
    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*20

    with open('/tmp/sav005/act.json', 'rb') as fp:
        traj = json.load(fp)
        tool_poses = np.asarray(traj['tool_pose'], dtype=np.float32)
        grasp_flags = np.asarray(traj['grasp_flag'], dtype=bool)
        print('grasp_flags', grasp_flags)

    to_grasp_command = []
    q_curr = env.get_joint_positions()
    for tool_pose, grasp_flag in tqdm(zip(tool_poses, grasp_flags), total=len(tool_poses)):
        xyz = tool_pose[..., :3, 3]
        rot = tool_pose[..., :3, :3]
        quat = (R.from_matrix(rot)).as_quat()
        q = np.concatenate([xyz, quat], axis=-1)
        try:
            q = env.solve_tool_ik(q, 'left',
                    check_collision=(not grasp_flag),
                    gripper_open=(not grasp_flag))
            q = np.concatenate([q, env.get_joint_positions()[6:]])
            # seems necessary to stabilize motion plan...?
            env.set_joint_positions(q) 
            to_grasp_command.append(Command(q, (not grasp_flag), False))
        except (AttributeError, ValueError):
            continue
    env.set_joint_positions(q_curr)

    #q_goal = np.concatenate([grasp_q_left, env.get_joint_positions()[6:]])
    #to_grasp_command = mp.get_joint_command(open_gripper_start=True, q_goal=q_goal, open_gripper=True)
    
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*20
    
    q_goal = np.concatenate([up_q_left, env.get_joint_positions()[6:]])
    to_up_command = mp.get_joint_command(open_gripper_start=False, q_start=to_grasp_command[-1].target_q, q_goal=q_goal, open_gripper=False, check_collision=False)

    command = gripper_open_command + to_grasp_command# + gripper_close_command + to_up_command
    
    obs_hist, imgs = env.execute_command(command, render=False, num_steps_after=100)

    print("success: ", env.check_success())

if __name__ == '__main__':
    main()
