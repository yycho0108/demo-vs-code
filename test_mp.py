from env.bimanual_bullet import PickCubeEnv
from mp.ompl_wrapper import MotionPlanner, Command


if __name__ == "__main__":
    cfg = {
        "sim_hz": 240,
        "control_hz": 10,
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

    # Joint goal motion planning
    env.reset()
    q_goal = [
        0.0, -0.5, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0, 0.0, 0.0
    ]
    command = mp.get_joint_command(q_goal=q_goal)

    imgs = env.execute_command(command, render=True)


    # EE goal motion planning
    env.reset()
    left_ee_pose, right_ee_pose = env.get_ee_pose()

    left_ee_pose[0] += 0.1
    left_ee_pose[2] += 0.1

    right_ee_pose[0] += 0.1
    right_ee_pose[2] += 0.1

    command1 = mp.get_joint_command(left_ee_goal=left_ee_pose)
    command2 = mp.get_joint_command(q_start=command1[-1].target_q, right_ee_goal=right_ee_pose)
    command = command1 + command2

    imgs = env.execute_command(command, render=True)

    # Gripper control
    env.reset()

    # Gripper open
    gripper_open_command = [Command(left_gripper_open=True, right_gripper_open=True)]*100
    # Gripper close
    gripper_close_command = [Command(left_gripper_open=False, right_gripper_open=False)]*100
    command = gripper_open_command + gripper_close_command

    imgs = env.execute_command(command)
