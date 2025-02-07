import numpy as np
from env.bimanual_bullet import CurlingEnv
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
            "gripper_gain": 0.05,
        },
        "problems": [0,1,2],
    }

    env = CurlingEnv(cfg)
    mp = MotionPlanner(env)

    env.reset()
    left_tool_pose, right_tool_pose = env.get_tool_poses()
    object_pose_dict = env.get_object_poses()
    curling_pose = object_pose_dict["curling"]

    # Solve IK for pre-defined poses
    target_left_tool_pose_approach = np.concatenate([curling_pose[:3], left_tool_pose[3:]])
    target_left_tool_pose_approach[2] += 0.18
    target_left_tool_pose_approach[0] += -0.03
    z_rot_candidates = np.linspace(0, -np.pi/2, 10)
    for z_rot in z_rot_candidates:
        target_left_tool_pose_approach[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(target_left_tool_pose_approach)
        q_approach = env.solve_tool_ik(target_left_tool_pose_approach, "left", gripper_open=True, check_collision=True)
        if q_approach is not None:
            break
    else:
        raise ValueError("No valid approach pose found")

    target_left_tool_pose_pre_push = np.concatenate([curling_pose[:3], left_tool_pose[3:]])
    target_left_tool_pose_pre_push[2] += 0.18
    z_rot_candidates = np.linspace(0, -np.pi/4, 10)
    for z_rot in z_rot_candidates:
        target_left_tool_pose_pre_push[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(target_left_tool_pose_pre_push)
        q_pre_push = env.solve_tool_ik(target_left_tool_pose_pre_push, "left", gripper_open=True, check_collision=True)
        if q_pre_push is not None:
            break
    else:
        raise ValueError("No valid pre-push pose found")
    
    target_left_tool_pose_post_push = np.concatenate([curling_pose[:3], left_tool_pose[3:]])
    target_left_tool_pose_post_push[0] += 0.1
    target_left_tool_pose_post_push[2] += 0.18
    z_rot_candidates = np.linspace(0, -np.pi/4, 10)
    for z_rot in z_rot_candidates:
        target_left_tool_pose_post_push[3:] = (R.from_quat(left_tool_pose[3:]) * R.from_euler('z', z_rot)).as_quat()
        env.draw_frame(target_left_tool_pose_post_push)
        q_post_push = env.solve_tool_ik(target_left_tool_pose_post_push, "left", check_collision=False)
        if q_post_push is not None:
            break
    else:
        raise ValueError("No valid post-push pose found")

    # Do motion planning
    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*20
 
    q_goal = np.concatenate([q_approach, env.get_joint_positions()[6:]])
    to_approach_command = mp.get_joint_command(
        open_gripper_start=True,
        q_goal=q_goal, 
        open_gripper=True,
        check_collision=True,
    )
    if not to_approach_command:
        raise ValueError("No valid pre-push trajectory found")
 
    q_goal = np.concatenate([q_pre_push, env.get_joint_positions()[6:]])
    to_pre_push_command = mp.get_joint_command(
        open_gripper_start=True,
        q_goal=q_goal, 
        open_gripper=True,
        check_collision=True,
    )
    if not to_pre_push_command:
        raise ValueError("No valid pre-push trajectory found")

    q_goal = np.concatenate([q_post_push, env.get_joint_positions()[6:]])
    to_post_push_command = mp.get_joint_command(
        q_start=to_pre_push_command[-1].target_q, 
        open_gripper_start=True, 
        q_goal=q_goal, 
        open_gripper=True,
        interpolation_res=0.1,
        check_collision=False,
    )
    if not to_post_push_command:
        raise ValueError("No valid post-push trajectory found")

    command = gripper_open_command + to_approach_command + to_pre_push_command + to_post_push_command
    
    obs_hist, imgs = env.execute_command(command, render=False, num_steps_after=300)
    print('success: ', env.check_success())
