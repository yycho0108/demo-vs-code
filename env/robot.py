from dataclasses import dataclass
from pybullet_planning.pybullet_tools.utils import *
    

@dataclass
class Command:
    target_q: np.ndarray = None
    left_gripper_open: bool = None
    right_gripper_open: bool = None


class Robot:
    def __init__(self, sim, robot_id):
        self.sim = sim
        self.robot_id = robot_id
        self._configure_joint_info(robot_id)
        # self._set_dynamics(robot_id)

    def _configure_joint_info(self, robot_id):
        self.num_joints = self._get_joints_num(robot_id)
        self.num_arm_joints = self.num_joints - 4

        self.all_joint_indices = self._get_joints_indices(robot_id)

        self.left_arm_joint_indices = self.all_joint_indices[:6]
        self.right_arm_joint_indices = self.all_joint_indices[8:14]
        self.arm_joint_indices = self.left_arm_joint_indices + self.right_arm_joint_indices

        self.left_gripper_joint_indices = self.all_joint_indices[6:8]
        self.right_gripper_joint_indices = self.all_joint_indices[14:]

        self.all_joint_limits = self._get_joint_limits(robot_id)
        self.left_arm_joint_limits = [self.all_joint_limits[0][:6], self.all_joint_limits[1][:6]]
        self.right_arm_joint_limits = [self.all_joint_limits[0][8:14], self.all_joint_limits[1][8:14]]
        self.arm_joint_limits = [
            self.all_joint_limits[0][:6] + self.all_joint_limits[0][8:14],
            self.all_joint_limits[1][:6] + self.all_joint_limits[1][8:14],
        ]
        self.left_gripper_joint_limits = [
            self.all_joint_limits[0][6:8], 
            self.all_joint_limits[1][6:8]
        ]
        self.right_gripper_joint_limits = [
            self.all_joint_limits[0][14:], 
            self.all_joint_limits[1][14:]
        ]
        self.left_gripper_close_pos = [0.0, 0.0]
        self.left_gripper_open_pos = [-1.0, 1.0]
        self.right_gripper_close_pos = [0.0, 0.0]
        self.right_gripper_open_pos = [-1.0, 1.0]

        self.left_gripper_close_vel = [5.0, -5.0]
        self.right_gripper_close_vel = [5.0, -5.0]
        self.left_gripper_open_vel = [-5.0, 5.0]
        self.right_gripper_open_vel = [-5.0, 5.0]

        self.left_tool_link = get_joint(self.robot_id, "left_tool_joint")
        self.right_tool_link = get_joint(self.robot_id, "right_tool_joint")

    def _set_dynamics(self, robot_id):
        for link_id in self.left_gripper_joint_indices + self.right_gripper_joint_indices:
            set_dynamics(
                robot_id,
                link_id,
                lateralFriction=0.5,
                spinningFriction=0.5,
                # rollingFriction=1.0,
                # restitution=0.0
            )

    def _get_joints_indices(self, robot_id):
        joints_indices = get_movable_joints(robot_id)
        return joints_indices

    def _get_joints_num(self, robot_id):
        num_joints=len(get_movable_joints(robot_id))
        return num_joints

    def _get_joint_limits(self, robot_id):
        joint_indices = self._get_joints_indices(robot_id)
        custom_limits_zip = get_custom_limits(robot_id, joint_indices)
        custom_limits = []
        for limit in custom_limits_zip:
            custom_limits.append(limit)
        return custom_limits
    
    def fk(self, q=None):
        assert len(q) == self.num_arm_joints, f"Given q {q} is not matched to num_joints {self.num_arm_joints}"

        q_orig = get_joint_positions(self.robot_id, self.arm_joint_indices)

        set_joint_positions(self.robot_id, self.arm_joint_indices, q)
        left_ee_pos, left_ee_quat = get_link_pose(self.robot_id, self.left_arm_joint_indices[-1])
        right_ee_pos, right_ee_quat = get_link_pose(self.robot_id, self.right_arm_joint_indices[-1])

        set_joint_positions(self.robot_id, self.arm_joint_indices, q_orig)

        return np.array(left_ee_pos + left_ee_quat + right_ee_pos + right_ee_quat)

    def ik(self, ee, left_or_right="left", max_attempts=5, max_iterations=200):
        """
        return None if no solution found
        """
        q = None
        if left_or_right == "left":
            qs = multiple_sub_inverse_kinematics(
                self.robot_id,
                self.left_arm_joint_indices[0],
                self.left_tool_link,
                (ee[:3], ee[3:]),
                max_attempts=max_attempts,
                max_iterations=max_iterations,
            )
            if len(qs) > 0:
                for q_sol in qs:
                    q_sol = np.array(q_sol[:6])
                    within_limits = ((self.left_arm_joint_limits[0] <= q_sol) & (q_sol <= self.left_arm_joint_limits[1])).all()
                    if within_limits:
                        q = np.array(q_sol[:6])
                        break
                else:
                    print("No IK solution within joint limits found for left arm")
        elif left_or_right == "right":
            qs = multiple_sub_inverse_kinematics(
                self.robot_id,
                self.right_arm_joint_indices[0],
                self.right_tool_link,
                (ee[:3], ee[3:]),
                max_attempts=max_attempts,
                max_iterations=max_iterations,
            )
            if len(qs) > 0:
                for q_sol in qs:
                    q_sol = np.array(q_sol[6:])
                    within_limits = ((self.right_arm_joint_limits[0] <= q_sol) & (q_sol <= self.right_arm_joint_limits[1])).all()
                    if within_limits:
                        q = np.array(q_sol[6:])
                        break
                else:
                    print("No IK solution within joint limits found for right arm")
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")

        return q