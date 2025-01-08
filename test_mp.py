from env.bimanual_bullet import PickCubeEnv
from mp.ompl_wrapper import MotionPlanner


if __name__ == "__main__":
    cfg = {
        "hz": 100,
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
    q_goal = [
        0.0, -0.5, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0, 0.0, 0.0
    ]
    traj = mp.get_joint_trajectory(q_goal=q_goal)

    env.reset()
    imgs = env.execute_trajectory(traj, render=True)


    # EE goal motion planning
    left_ee_pose, right_ee_pose = env.get_ee_pose()

    left_ee_pose[0] += 0.1
    left_ee_pose[2] += 0.1

    right_ee_pose[0] += 0.1
    right_ee_pose[2] += 0.1

    traj1 = mp.get_joint_trajectory(left_ee_goal=left_ee_pose)
    traj2 = mp.get_joint_trajectory(q_start=traj1[-1], right_ee_goal=right_ee_pose)

    traj = traj1 + traj2

    env.reset()
    imgs = env.execute_trajectory(traj, render=True)
