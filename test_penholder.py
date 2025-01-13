import numpy as np
from env.bimanual_bullet import PenholderEnv
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

    env = PenholderEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    left_ee_pose, right_ee_pose = env.get_ee_pose()
    object_pose_dict = env.get_object_poses()
    pen_pose = object_pose_dict["pen"]
    
    # target_left_ee_pos = pen_pose[:3].copy()-[0.05, 0., 0.]
    # target_left_ee_quat = (R.from_quat(left_ee_pose[3:]) * R.from_euler('z', -30, degrees=True)).as_quat()
    # target_left_ee_pose = np.concatenate([target_left_ee_pos, target_left_ee_quat])

    target_left_ee_pose = np.array([0.30906999,  0.13185944,  0.60955596, -0.03678786,  0.0999244 , -0.80908298 ,0.57796752])
    # target_q = np.array([-0.336, -0.056, -0.276, 0.482, 0.0, 0.0])

    # env.set_single_arm_joint_positions(target_q)
    # print(env.get_ee_pose()[0])

    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*50
    to_box_command = mp.get_joint_command(left_ee_goal=target_left_ee_pose, open_gripper=True)
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*50
    target_left_ee_pose[2] += 0.1
    to_up_command = mp.get_joint_command(q_start=to_box_command[-1].target_q, left_ee_goal=target_left_ee_pose, open_gripper=False)

    command = gripper_open_command + to_box_command + gripper_close_command + to_up_command
    
    imgs = env.execute_command(command, render=True)
    print('success: ', env.check_success())
