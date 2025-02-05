import numpy as np
from env.bimanual_bullet import CurlingEnv
from mp.ompl_wrapper import MotionPlanner, Command
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    cfg = {
        "gui": True,
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

    env = CurlingEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    left_ee_pose, right_ee_pose = env.get_ee_pose()
    object_pose_dict = env.get_object_poses()
    curling_pose = object_pose_dict["curling"]

    grasp_pos = curling_pose[:3].copy()
    grasp_pos[2] += 0.15

    env.draw_points(grasp_pos, size=20)
    
    target_left_tool_pose = np.array([7.47078806e-02,  2.79245913e-01,  6.21249437e-01, -6.61014095e-02,  1.31183490e-01, -6.77508175e-01, 7.20697045e-01])
    # target_q = np.array([0.294, -0.165, 0.369, -0.124, 0.0, -0.253])

    # env.set_single_arm_joint_positions(target_q)
    # print(env.get_ee_pose()[0])

    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*50
    to_box_command = mp.get_joint_command(left_tool_goal=target_left_tool_pose, open_gripper=True)
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*50
    target_left_tool_pose[1] += 0.1
    to_up_command = mp.get_joint_command(q_start=to_box_command[-1].target_q, left_tool_goal=target_left_tool_pose, open_gripper=False)

    command = gripper_open_command + to_box_command + gripper_close_command + to_up_command + gripper_open_command
    
    imgs = env.execute_command(command, render=True)
    print('success: ', env.check_success())
