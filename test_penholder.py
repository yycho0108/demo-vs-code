import numpy as np
from env.bimanual_bullet import PenholderEnv
from mp.ompl_wrapper import MotionPlanner, Command
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
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

    env = PenholderEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    left_tool_pose, right_tool_pose = env.get_tool_pose()
    object_pose_dict = env.get_object_poses()
    pen_pose = object_pose_dict["pen"]
    holder_pose = object_pose_dict["holder"]

    grasp_left_tool_pose = np.concatenate([pen_pose[:3], left_tool_pose[3:]])
    z_rot_candidates = np.linspace(-np.pi/2, 0, 10)
    for z_rot in z_rot_candidates:
        grasp_left_tool_pose[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(grasp_left_tool_pose)
        grasp_q_left = env.solve_tool_ik(grasp_left_tool_pose, "left", check_collision=True)
        if grasp_q_left is not None:
            break
    else:
        raise ValueError("No valid grasp pose found")
    
    up_grasp_left_tool_pose = np.concatenate([pen_pose[:3], left_tool_pose[3:]])
    up_grasp_left_tool_pose[2] += 0.15
    z_rot_candidates = np.linspace(-np.pi/2, 0, 10)
    for z_rot in z_rot_candidates:
        up_grasp_left_tool_pose[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(up_grasp_left_tool_pose)
        up_grasp_q_left = env.solve_tool_ik(up_grasp_left_tool_pose, "left", check_collision=True)
        if up_grasp_q_left is not None:
            break
    else:
        raise ValueError("No valid up pose found")
    
    up_holder_left_tool_pose = np.concatenate([holder_pose[:3], left_tool_pose[3:]])
    up_holder_left_tool_pose[2] += 0.19
    z_rot_candidates = np.linspace(-np.pi, 0, 10)
    for z_rot in z_rot_candidates:
        up_holder_left_tool_pose[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(up_holder_left_tool_pose)
        up_holder_q_left = env.solve_tool_ik(up_holder_left_tool_pose, "left", check_collision=True)
        if up_holder_q_left is not None:
            break
    else:
        raise ValueError("No valid up pose found")

    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*20

    q_goal = np.concatenate([grasp_q_left, env.get_joint_positions()[6:]])
    to_box_command = mp.get_joint_command(open_gripper_start=True, q_goal=q_goal, open_gripper=True)

    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*20
    
    q_goal = np.concatenate([up_grasp_q_left, env.get_joint_positions()[6:]])
    to_up_grasp_command = mp.get_joint_command(q_start=to_box_command[-1].target_q, open_gripper_start=False, q_goal=q_goal, open_gripper=False, check_collision=False)

    q_goal = np.concatenate([up_holder_q_left, env.get_joint_positions()[6:]])
    to_up_holder_command = mp.get_joint_command(q_start=to_up_grasp_command[-1].target_q, open_gripper_start=False, q_goal=q_goal, open_gripper=False, check_collision=False)

    command = gripper_open_command + to_box_command + gripper_close_command + to_up_grasp_command + to_up_holder_command + gripper_open_command
    
    imgs = env.execute_command(command, render=False)
    
    print('success: ', env.check_success())
