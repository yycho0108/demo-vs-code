import numpy as np
from env.bimanual_bullet import PickCubeEnv
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

    env = PickCubeEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    joint_positions = env.get_joint_positions()
    left_tool_pose, right_tool_pose = env.get_tool_poses()
    object_pose_dict = env.get_object_poses()
    cube_pose = object_pose_dict["cube"]
    env.draw_frame(left_tool_pose)
    env.draw_frame(right_tool_pose)

    # Solve IK for pre-defined poses
    grasp_left_tool_pos = cube_pose[:3]
    grasp_left_tool_quat = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', -np.pi/4)).as_quat()
    grasp_left_tool_pose = np.concatenate([grasp_left_tool_pos, grasp_left_tool_quat])
    env.draw_frame(grasp_left_tool_pose)
    grasp_q_left = env.solve_tool_ik(grasp_left_tool_pose, "left", gripper_open=True, check_collision=True)
    if grasp_q_left is None:
        raise ValueError("No valid grasp pose found")

    # Do motion planning
    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*20

    q_goal = np.concatenate([grasp_q_left, env.get_joint_positions()[6:]])
    to_grasp_command = mp.get_joint_command(open_gripper_start=True, q_goal=q_goal, open_gripper=True)
    
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*20

    command = gripper_open_command + to_grasp_command + gripper_close_command
    
    obs_hist, imgs = env.execute_command(command, render=False, num_steps_after=100)

    print("success: ", env.check_success())

