import numpy as np
from env.bimanual_bullet import PickCubeEnv
from mp.ompl_wrapper import MotionPlanner, Command
from scipy.spatial.transform import Rotation as R
import json
from tqdm.auto import tqdm


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
        # ==================================================
        # == CHANGE PROBLEM INDEX in [0,1,2] TO TEST ======
        # ==================================================
        "problems": [0],
        # ==================================================
        # ==================================================
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

    # ==================================================
    # == CHANGE FILENAMES & LOADING LOGIC ==============
    # ==================================================
    with open('/tmp/sav011/act.json', 'rb') as fp:
        traj = json.load(fp)
        tool_poses = np.asarray(traj['tool_pose'], dtype=np.float32)
        grasp_flags = np.asarray(traj['grasp_flag'], dtype=bool)
    # ==================================================
    # ==================================================

    commands = []
    q_curr = env.get_joint_positions()
    for tool_pose, grasp_flag in tqdm(
            zip(tool_poses, grasp_flags),
            total=len(tool_poses)):

        # ==================================================
        # == REPLACE THIS SECTION (AND others, if needed) ==
        LEFT_TOOL = env.get_tool_poses()[0] # WARN(ycho): PLACEHOLDER!
        xyz = LEFT_TOOL[..., 0:3]
        quat = LEFT_TOOL[..., 3:7]
        grasp_flag = False
        # ==================================================
        # ==================================================

        target_pose = np.concatenate([xyz, quat], axis=-1)
        try:
            q = env.solve_tool_ik(target_pose, 'left',
                                  check_collision=(not grasp_flag),
                                  gripper_open=(not grasp_flag))
            q = np.concatenate([q, env.get_joint_positions()[6:]])
            env.set_joint_positions(q)
            commands.append(Command(q, (not grasp_flag), False))
        except (AttributeError, ValueError):
            continue
    env.set_joint_positions(q_curr)

    # ==================================================
    # == REPLACE THIS SECTION (AND others, if needed) ==
    commands = (
        ([commands[0]] * 10)
        + commands
        + ([commands[-1]] * 10)
    )
    # ==================================================
    # ==================================================

    obs_hist, imgs = env.execute_command(commands, render=False, num_steps_after=100)
    print("success: ", env.check_success())
