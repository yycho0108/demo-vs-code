import numpy as np
from env.bimanual_bullet import PickCubeEnv
from mp.ompl_wrapper import MotionPlanner, Command
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    cfg = {
        "sim_hz": 240,
        "control_hz": 240,
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
        }
    }

    env = PickCubeEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    left_ee_pose, right_ee_pose = env.get_ee_pose()
    object_pose_dict = env.get_object_poses()
    cube_pose = object_pose_dict["cube"]
    
    target_left_ee_pos = cube_pose[:3].copy()
    target_left_ee_quat = (R.from_quat(left_ee_pose[3:]) * R.from_euler('z', -45, degrees=True)).as_quat()
    target_left_ee_pose = np.concatenate([target_left_ee_pos, target_left_ee_quat])

    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*50
    to_box_command = mp.get_joint_command(left_ee_goal=target_left_ee_pose, open_gripper=True)
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*50
    target_left_ee_pose[2] += 0.1
    to_up_command = mp.get_joint_command(q_start=to_box_command[-1].target_q, left_ee_goal=target_left_ee_pose, open_gripper=False)

    command = gripper_open_command + to_box_command + gripper_close_command + to_up_command
    
    imgs = env.execute_command(command, render=True)
