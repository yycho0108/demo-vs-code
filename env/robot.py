from pybullet_planning.pybullet_tools.utils import *


class Robot:
    def __init__(self, sim, robot_id):
        self.sim = sim
        self.robot_id = robot_id

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
        self.right_arm_joint_limits = [self.all_joint_limits[0][8:14] + self.all_joint_limits[1][8:14]]
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
        left_ee = get_link_pose(self.robot_id, self.robot.left_arm_joint_indices[-1])
        right_ee = get_link_pose(self.robot_id, self.robot.right_arm_joint_indices[-1])

        set_joint_positions(self.robot_id, self.arm_joint_indices, q_orig)

        return np.array([left_ee, right_ee])

    def ik(self, ee, left_or_right="left"):
        """
        return None if no solution found
        """
        q = None
        if left_or_right == "left":
            qs = multiple_sub_inverse_kinematics(self.robot_id, self.left_arm_joint_indices[0], self.left_arm_joint_indices[-1], (ee[:3], ee[3:]))
            if len(qs) > 0 :
                q = qs[0][:6]
        elif left_or_right == "right":
            qs = multiple_sub_inverse_kinematics(self.robot_id, self.right_arm_joint_indices[0], self.right_arm_joint_indices[-1], (ee[:3], ee[3:]))
            if len(qs) > 0:
                q = qs[0][8:14]
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")
        return np.array(q)