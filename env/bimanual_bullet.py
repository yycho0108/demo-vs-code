from pybullet_utils import bullet_client
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
import pybullet_data
import numpy as np
import pybullet
import time
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))


from pybullet_planning.pybullet_tools.utils import *
try:
    from .robot import Robot
except:
    from robot import Robot


def getJointStates(robot):
  joint_states = sim.getJointStates(robot, range(sim.getNumJoints(robot)))
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
  joint_states = sim.getJointStates(robot, range(sim.getNumJoints(robot)))
  joint_infos = [sim.getJointInfo(robot, i) for i in range(sim.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def ompl_state_to_list(state, num_joints):
    return [state[i] for i in range(num_joints)]


def get_validate_fn(body, joints, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                     custom_limits={}, use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_links = frozenset(link for link in get_moving_links(body, joints)
                             if can_collide(body, link))
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(map(parse_body, attached_bodies))
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
    limits_fn = get_limits_fn(body, joints, custom_limits=custom_limits)

    def validate_fn(state):
        q = ompl_state_to_list(state, len(joints))
        if limits_fn(q):
            return False
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

        for link1, link2 in check_link_pairs:
            if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                    pairwise_link_collision(body, link1, body, link2):
                return False

        for body1, body2 in product(moving_bodies, obstacles):
            if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    and pairwise_collision(body1, body2, **kwargs):
                return False
        return True
    return validate_fn

def create_frame(radius=0.005, height=0.1):
    x = create_cylinder(radius=radius, height=height, color=RED, collision=False)
    y = create_cylinder(radius=radius, height=height, color=GREEN, collision=False)
    z = create_cylinder(radius=radius, height=height, color=BLUE, collision=False)
    return x, y, z

def get_axes_poses(pose, height):
    x_pose = p.multiplyTransforms(
        positionA=pose[:3],
        orientationA=pose[3:],
        positionB=np.array([height/2,0,0]),
        orientationB=R.from_rotvec([0,np.pi/2,0]).as_quat(),
    )

    y_pose = p.multiplyTransforms(
        positionA=pose[:3],
        orientationA=pose[3:],
        positionB=np.array([0,height/2,0]),
        orientationB=R.from_rotvec([-np.pi/2,0,0]).as_quat(),
    )

    z_pose = p.multiplyTransforms(
        positionA=pose[:3],
        orientationA=pose[3:],
        positionB=np.array([0,0,height/2]),
        orientationB=np.array([0,0,0,1]),
    )
    
    return x_pose, y_pose, z_pose


class EnvBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot_id = None
        self.obs_ids = {}
        self.obj_ids = {}
        self.debug_ids = []
        self.set_seed(cfg["seed"])
        self._load_plane_and_robot(cfg)
        self._set_render_cfg(cfg["render"])

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)

    def _load_plane_and_robot(self, cfg):
        # Set cfg
        self.gui = cfg["gui"]
        self.sim_hz = cfg["sim_hz"]
        self.ctrl_step = cfg["sim_hz"] // cfg["control_hz"]
        self.dt = 1 / self.sim_hz
        self.gripper_gain = cfg["ctrl"]["gripper_gain"]

        if self.gui:
            connection_mode = pybullet.GUI
        else:
            connection_mode = pybullet.DIRECT

        self.sim = bullet_client.BulletClient(connection_mode=connection_mode)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        self.sim.setGravity(0, 0, -9.81)
        self.sim.setTimeStep(self.dt)

        # Plane
        plane_id = self.sim.loadURDF("plane.urdf")
        self.obs_ids["plane"] = plane_id

        # Panda
        start_pos = [0, 0, 0]
        start_quat = self.sim.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = self.sim.loadURDF('assets/RobotBimanualV4/urdf/RobotBimanualV4_gripper.urdf', start_pos, start_quat, useFixedBase=True, globalScaling=1., flags=self.sim.URDF_USE_SELF_COLLISION)
        self.robot = Robot(self.sim, self.robot_id)
        self.set_joint_positions(cfg["q_init"])
        # self.set_gripper_open("left")
        # self.set_gripper_open("right")

    @abstractmethod
    def loadEnv(self):
        pass

    @abstractmethod
    def reset():
        pass

    def get_joint_positions(self):
        """
        Get current joint positions.

        Args:

        Returns:
            q (np.ndarray): current joint positions (size 12)

        """
        return np.array(get_joint_positions(self.robot_id, self.robot.arm_joint_indices))
    
    def set_joint_positions(self, q):
        """
        Set joint positions.

        Args:
            q (np.ndarray): joint positions to set (size 12)

        Returns:

        """
        assert len(q) == self.robot.num_arm_joints, f"Given q {q} is not matched to num_joints {self.robot.num_arm_joints}"
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q)
    
    def set_single_arm_joint_positions(self, q, left_or_right="left"):
        """
        Set joint positions.

        Args:
            q (np.ndarray): single arm joint positions to set (size 6)
            left_or_right (str): "left" or "right" to set

        Returns:

        """
        assert left_or_right in ["left", "right"], "left_or_right should be either 'left' or 'right'"
        assert len(q) == self.robot.num_arm_joints//2, f"Given q {q} is not matched to num_joints {self.robot.num_arm_joints//2}"
        
        q_all = self.get_joint_positions()
        if left_or_right == "left":
            q_all[:6] = q
        elif left_or_right == "right":
            q_all[6:] = q
        
        self.set_joint_positions(q_all)
    
    def get_ee_pose(self):
        """
        Get current pose of end-effector frame.

        Args:

        Returns:
            pose (np.ndarray): pose of end-effector frame. (position (3) + quaternion (4))

        """
        left_ee_pos, left_ee_quat = get_link_pose(self.robot_id, self.robot.left_arm_joint_indices[-1])
        right_ee_pos, right_ee_quat = get_link_pose(self.robot_id, self.robot.right_arm_joint_indices[-1])
        return np.array([left_ee_pos + left_ee_quat, right_ee_pos + right_ee_quat])
    
    def get_tool_pose(self):
        """
        Get current pose of tool frame.

        Args:

        Returns:
            pose (np.ndarray): pose of tool frame. (position (3) + quaternion (4))

        """
        left_tool_pos, left_tool_quat = get_link_pose(self.robot_id, self.robot.left_tool_link)
        right_tool_pos, right_tool_quat = get_link_pose(self.robot_id, self.robot.right_tool_link)
        return np.array([left_tool_pos + left_tool_quat, right_tool_pos + right_tool_quat])
    
    def get_gripper_joint_position(self):
        """
        Get current gripper joint position.

        Args:

        Returns:
            left_finger_pos (np.ndarray): left gripper joint position (size 2)
            right_finger_pos (np.ndarray): right gripper joint position (size 2)

        """
        left_finger_pos = np.array(get_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices))
        right_finger_pos = np.array(get_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices))
        return left_finger_pos, right_finger_pos

    def get_gripper_opened(self):
        """
        Get current gripper opened.

        Args:

        Returns:
            gripper_opened (np.ndarray): boolean of left and right gripper open or not (size 2)

        """
        left_finger_pos, right_finger_pos = self.get_gripper_joint_position()
        is_left_opened = np.diff(left_finger_pos) > (self.robot.left_gripper_joint_limits[1][1] - self.robot.left_gripper_joint_limits[0][0])/2
        is_right_opened = np.diff(right_finger_pos) > (self.robot.right_gripper_joint_limits[1][1] - self.robot.right_gripper_joint_limits[0][0])/2

        return np.concatenate([is_left_opened, is_right_opened])
    
    def set_gripper_open(self, left_or_right="left"):
        """
        Set gripper open.

        Args:

        Returns:
            left_or_right (str): left or right gripper.

        """
        if left_or_right == "left":
            set_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices, self.robot.left_gripper_open_pos)
        elif left_or_right == "right":
            set_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices, self.robot.right_gripper_open_pos)
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")
        
    def set_gripper_close(self, left_or_right="left"):
        """
        Set gripper close.

        Args:

        Returns:
            left_or_right (str): left or right gripper.

        """
        if left_or_right == "left":
            set_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices, self.robot.left_gripper_close_pos)
        elif left_or_right == "right":
            set_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices, self.robot.right_gripper_close_pos)
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")
    
    def get_object_poses(self):
        """
        Get object poses.

        Args:

        Returns:
            pose_dict (dict): dictionary of object name (str) and its pose (np.ndarray: position (3) + quaternion (4))

        """
        pose_dict = {}
        for obs_name, obs_id in self.obs_ids.items():
            pos, quat = get_pose(obs_id)
            pose_dict[obs_name] = np.array(list(pos) + list(quat))

        for obj_name, obj_id in self.obj_ids.items():
            pos, quat = get_pose(obj_id)
            pose_dict[obj_name] = np.array(list(pos) + list(quat))
        return pose_dict
    
    def check_collision(self, q):
        """
        Check collision for given joint positions.

        Args:
            q (np.ndarray): joint positions to check (size 12)

        Returns:
            is_collision (bool): True if collision. False if not.

        """
        q_orig = get_joint_positions(self.robot_id, self.robot.arm_joint_indices)
        is_collision = get_collision_fn(
            body=self.robot_id,
            joints=self.robot.arm_joint_indices,
            obstacles=self.obs_ids.values(),
            cache=True
        )(q)
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q_orig)
        return is_collision
    
    def solve_tool_ik(self, tool_pose, left_or_right="left", max_attempts=5, check_collision=True):
        """
        Solve inverse kinematics (IK) for given tool pose.

        Args:
            tool_pose (np.ndarray): tool pose (position (3) + quaternion (4)) to solve IK.
            left_or_right (str): which arm to solve IK among "left" and "right".
            max_attempts (int): (optional) max attempts to solve IK.
            check_collision (bool): (optional) check collision for solved IK or not.
            
        Returns:
            q (np.ndarray): solution of IK (size 6)

        """
        assert left_or_right in ["left", "right"], "left_or_right should be either 'left' or 'right'"
        q = self.get_joint_positions()
        q_ik = self.robot.ik(tool_pose, left_or_right, max_attempts=max_attempts)
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q)
        
        if q_ik is None:
            print("No IK solution is found!!")
            return None
        
        if not check_collision:
            return q_ik
        
        if left_or_right == "left":
            q[:6] = q_ik
        elif left_or_right == "right":
            q[6:] = q_ik
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")

        if self.check_collision(q):
            print("IK solution is in collision!!")
            return None

        return q_ik

    def _get_val_fn(self):
        val_fn = get_validate_fn(
            body=self.robot_id,
            joints=self.robot.arm_joint_indices,
            obstacles=list(self.obs_ids.values()) + list(self.obj_ids.values()),
            cache=True
        )
        return val_fn

    def _get_attached_val_fn(self, attached_obj_name):
        attach_val_fn = get_validate_fn(
            body=self.robot_id,
            joints=self.robot.arm_joint_indices,
            obstacles=self.obs_ids.values(),
            attachments=[self.obj_ids[attached_obj_name]],
            cache=True,
        )
        return attach_val_fn
    
    def _set_target_joint_position(self, command):
        if command.target_q is None:
            return
        self.sim.setJointMotorControlArray(
            self.robot_id,
            self.robot.arm_joint_indices,
            self.sim.POSITION_CONTROL,
            targetPositions=command.target_q,
            # positionGains=[0.2]*self.robot.num_arm_joints
        )

    def _set_gripper_command(self, command):
        if command.left_gripper_open is not None:
            if command.left_gripper_open:
                target_left_gripper_pos = self.robot.left_gripper_open_pos
            else:
                target_left_gripper_pos = self.robot.left_gripper_close_pos

            self.sim.setJointMotorControlArray(
                self.robot_id, 
                self.robot.left_gripper_joint_indices, 
                self.sim.POSITION_CONTROL,
                targetPositions=target_left_gripper_pos,
                positionGains=[self.gripper_gain]*2,
            )

        if command.right_gripper_open is not None:
            if command.right_gripper_open:
                target_right_gripper_pos = self.robot.right_gripper_open_pos
            else:
                target_right_gripper_pos = self.robot.right_gripper_close_pos

            self.sim.setJointMotorControlArray(
                self.robot_id, 
                self.robot.right_gripper_joint_indices, 
                self.sim.POSITION_CONTROL,
                targetPositions=target_right_gripper_pos,
                positionGains=[self.gripper_gain]*2,
            )

    def execute_command(self, command, render=False, num_steps_after=0):
        """
        Simulate the robot for given commands.
        
        Args:
            command (list[Command]): list of command to execute the robot.
            render (bool): render images or not.
            num_steps_after (int): number of steps to simulate after executing the command.

        Returns:
            imgs (list): list of images (empty if render is False)

        """
        imgs = []
        for i in tqdm(range(len(command))):
            for _ in range(self.ctrl_step):
                self._set_target_joint_position(command[i])
                self._set_gripper_command(command[i])
                self.sim.stepSimulation()

                if not render:
                    time.sleep(self.dt)
            
            if render:
                imgs.append(self.render())
    
        for i in range(num_steps_after):
            for _ in range(self.ctrl_step):
                self.sim.stepSimulation()

                if not render:
                    time.sleep(self.dt)

            if render:
                imgs.append(self.render())

        return imgs

    def _set_render_cfg(self, render_cfg):
        self.width, self.height = render_cfg["width"], render_cfg["height"]

        # Capture the current frame
        self.view_matrix = self.sim.computeViewMatrix(
            cameraEyePosition=render_cfg["cam"]["pos"],
            cameraTargetPosition=render_cfg["cam"]["lookat"],
            cameraUpVector=render_cfg["cam"]["up"],
        )
        self.projection_matrix = self.sim.computeProjectionMatrixFOV(
            fov=render_cfg["cam"]["fov"], aspect=float(self.width) / self.height, nearVal=0.1, farVal=100.0
        )

    def render(self):
        img = self.sim.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix)
        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]  # Extract RGB data
        return rgb

    def clear_vis(self):
        """
        Remove all visualization items.

        Args:

        Returns:

        """

        for id in self.debug_ids:
            self.sim.removeBody(id)

        self.debug_ids = []

        # self.sim.removeAllUserParameters()

    def draw_points(self, positions, rgba=None, radius=0.01):
        """
        Draw points in a simulation.

        Args:
            positions (np.ndarray): positions of points (3 or N x 3)
            rgba (np.ndarray): rgba (0 ~ 1) of points (N x 4)
            size (float): radius of points

        Returns:

        """

        assert positions.shape[-1] == 3, "Invalid shape of positions"

        if positions.ndim == 1:
            positions = np.expand_dims(positions, axis=0)

        if rgba is None:
            rgba = np.zeros((positions.shape[0], 4))
            rgba[:,0] = 1
            rgba[:,3] = 1

        for pos, color in zip(positions, rgba):
            id = create_sphere(radius=radius, color=color, collision=False)
            set_position(id, pos[0], pos[1], pos[2])
            self.debug_ids.append(id)

        # self.sim.addUserDebugPoints(
        #     pointPositions=positions, 
        #     pointColorsRGB=colors, 
        #     pointSize=size,
        # )

    def draw_frame(self, pose, radius=0.005, height=0.1):
        """
        Draw frame in a simulation.

        Args:
            pose (np.ndarray): pose of frame (position (3) + quaternion (4))
            size (float): size of frame

        Returns:

        """

        x_id, y_id, z_id = create_frame(radius, height)
        x_pose, y_pose, z_pose = get_axes_poses(pose, height)

        set_pose(x_id, x_pose)
        set_pose(y_id, y_pose)
        set_pose(z_id, z_pose)

        self.debug_ids.extend([x_id, y_id, z_id])


class PickCubeEnv(EnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loadEnv()

    def loadEnv(self):
        # Table
        table_shape = [0.5, 0.5, 0.8]
        table_pos = [0.4, 0.0, 0.2]
        table_quat = [0.0, 0.0, 0.0, 1.0]
        table_id = create_box(*table_shape)
        set_pose(table_id,(table_pos, table_quat))
        self.obs_ids["table"] = table_id

        # Cube
        cube_size = [0.015, 0.015, 0.1]
        card_z = table_pos[2] + table_shape[2]/2 + cube_size[2]/2
        self.cube_pos = [table_pos[0], 0.0, card_z]
        self.cube_quat = [0.0, 0.0, 0.0, 1.0]
        cube_id = create_box(*cube_size, mass=0.5, color=WHITE)
        set_pose(cube_id,(self.cube_pos, self.cube_quat))
        set_dynamics(cube_id, lateralFriction=100.0, spinningFriction=100.0, rollingFriction=0.01, restitution=0.0, contactStiffness=10000000.0, contactDamping=10000.0)
        self.obj_ids["cube"] = cube_id

    def reset(self):
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, self.cfg["q_init"])
        set_pose(self.obj_ids["cube"], (self.cube_pos, self.cube_quat))

    def check_success(self):
        cube_pos = self.get_object_poses()["cube"][:3]
        return cube_pos[2] > 0.7

class PenholderEnv(EnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loadEnv()

    def loadEnv(self):
        # Table
        table_shape = [0.4, 0.84, 0.6]
        table_pos = [0.3, 0.0, 0.3]
        table_quat = [0.0, 0.0, 0.0, 1.0]
        table_id = create_box(*table_shape, color=GREY)
        set_pose(table_id,(table_pos, table_quat))
        self.obs_ids["table"] = table_id

        # Cube
        holder_z = table_pos[2] + table_shape[2]/2
        self.holder_pos = [table_pos[0], 0.0, holder_z]
        self.holder_quat = [0.0, 0.0, 0.0, 1.0]
        penholder_id = create_obj(path="assets/meshes/penholder_coacd.obj", color=GREEN)
        set_pose(penholder_id,(self.holder_pos, self.holder_quat))
        self.obs_ids["holder"] = penholder_id

        # pen_radius = 0.01
        # pen_height = 0.12

        # pen_z = table_pos[2] + table_shape[2]/2 + pen_height/2
        # self.pen_pos = [table_pos[0]+0.1, 0.1, pen_z]
        # self.pen_quat = [0.0, 0.0, 0.0, 1.0]
        # pen_id = create_cylinder(pen_radius, pen_height, mass=0.1, color=BLUE)
        # set_pose(pen_id,(self.pen_pos, self.pen_quat))
        # self.obj_ids["pen"] = pen_id

        pen_size = [0.01, 0.01, 0.12]

        pen_z = table_pos[2] + table_shape[2]/2 + pen_size[2]/2
        self.pen_pos = [table_pos[0], 0.1, pen_z]
        self.pen_quat = [0.0, 0.0, 0.0, 1.0]
        pen_id = create_box(*pen_size, mass=0.1, color=BLUE)
        set_pose(pen_id,(self.pen_pos, self.pen_quat))
        set_dynamics(pen_id, lateralFriction=100.0, spinningFriction=100.0, rollingFriction=0.01, restitution=0.0, contactStiffness=10000000.0, contactDamping=10000.0)
        self.obj_ids["pen"] = pen_id

    def check_success(self):
        pen_pos = self.get_object_poses()["pen"][:3]
        holder_pos = self.get_object_poses()["holder"][:3]
        
        return np.linalg.norm(holder_pos[:2] - pen_pos[:2]) < 0.03 and holder_pos[2] < 0.8

    def reset(self):
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, self.cfg["q_init"])
        set_pose(self.obs_ids["holder"], (self.holder_pos, self.holder_quat))
        set_pose(self.obj_ids["pen"],(self.pen_pos, self.pen_quat))

class CurlingEnv(EnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loadEnv()

    def loadEnv(self):
        # Table
        table_shape = [2.0, 0.6, 0.6]
        table_pos = [1.1, 0.0, 0.2]
        table_quat = [0.0, 0.0, 0.0, 1.0]
        table_id = create_box(*table_shape, color=GREY)
        set_pose(table_id,(table_pos, table_quat))
        set_dynamics(table_id, lateralFriction=0.1, spinningFriction=0.1)
        self.obs_ids["table"] = table_id

        # Curling
        curling_z = table_pos[2] + table_shape[2]/2
        self.curling_pos = [0.4, 0.0, curling_z]
        self.curling_quat = [0, 0, 0.7071068, 0.7071068]
        curling_id = create_obj(scale=1.5, mass=1.0, path="assets/meshes/curling_high_handle_coacd.obj", color=BLUE)
        set_pose(curling_id,(self.curling_pos, self.curling_quat))
        set_dynamics(curling_id, lateralFriction=0.05, spinningFriction=0.05)
        self.obj_ids["curling"] = curling_id
    
    def reset(self):
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, self.cfg["q_init"])
        set_pose(self.obj_ids["curling"], (self.curling_pos, self.curling_quat))

    def check_success(self):
        curling_pos = self.get_object_poses()["curling"][:3]
        return curling_pos[0] > 1.0
    
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
    sim = env.sim

    L_JOINT_LIMIT_MIN = np.pi/180.*np.array([-90.,  -90., -180.,  -45., -210., -125.]) 
    L_JOINT_LIMIT_MAX = np.pi/180.*np.array([50.,  80., 80.,  75., 210. , 125.])
    R_JOINT_LIMIT_MIN = np.pi/180.*np.array([-50.,  -80., -80.,  -75., -210., -125.]) 
    R_JOINT_LIMIT_MAX = np.pi/180.*np.array([90.,  90., 180.,  45., 210. , 125.])

    numJoints = sim.getNumJoints(env.robot_id)
    link_id_dict = dict()
    joint_id_dict = dict()
    for _id in range(numJoints):
        joint_info = sim.getJointInfo(env.robot_id, _id)
        link_name = joint_info[12].decode('UTF-8')
        link_id_dict[link_name] = _id
        joint_name = joint_info[1].decode('UTF-8')
        joint_id_dict[joint_name] = _id
        print(link_name, joint_name, _id)

    control_dt = 1./100
    sim.setTimestep = control_dt

    
    debugparams = []
    MODE = 'pos' # 'jp' or 'pos' or 'sinusoidal' or 'inv_dyn'

    if MODE == 'pos':
        init_q = np.zeros(6)
        for i in range(6):
            debugparams.append(sim.addUserDebugParameter(f"theta_{i+1}",L_JOINT_LIMIT_MIN[i],L_JOINT_LIMIT_MAX[i],init_q[i]))
        
        for i in range(6, 12):
            debugparams.append(sim.addUserDebugParameter(f"theta_{i+1}",R_JOINT_LIMIT_MIN[i-6],R_JOINT_LIMIT_MAX[i-6],0))
        
        debugparams.append(sim.addUserDebugParameter(f"left_finger", -45.*np.pi/180., 0, -45.*np.pi/180.))
        debugparams.append(sim.addUserDebugParameter(f"right_finger", -45.*np.pi/180., 0, -45.*np.pi/180.))
            

    sim.setRealTimeSimulation(False)
    sim.setGravity(0, 0, -9.81)

    Start_Simul = True
    while Start_Simul:
        sim.stepSimulation()

        sim.setTimeStep(control_dt)

        thetas = []
        for param in debugparams:
           thetas.append(sim.readUserDebugParameter(param))
        
        if MODE == 'pos':
            robotId = env.robot_id
            sim.resetJointState(robotId, joint_id_dict['joint1'], targetValue=thetas[0])
            sim.resetJointState(robotId, joint_id_dict['joint2'], targetValue=thetas[1])
            sim.resetJointState(robotId, joint_id_dict['joint3'], targetValue=thetas[2])
            sim.resetJointState(robotId, joint_id_dict['joint4'], targetValue=thetas[3])
            sim.resetJointState(robotId, joint_id_dict['joint5'], targetValue=thetas[4])
            sim.resetJointState(robotId, joint_id_dict['joint6'], targetValue=thetas[5])

            sim.resetJointState(robotId, joint_id_dict['joint7'], targetValue=thetas[6])
            sim.resetJointState(robotId, joint_id_dict['joint8'], targetValue=thetas[7])
            sim.resetJointState(robotId, joint_id_dict['joint9'], targetValue=thetas[8])
            sim.resetJointState(robotId, joint_id_dict['joint10'], targetValue=thetas[9])
            sim.resetJointState(robotId, joint_id_dict['joint11'], targetValue=thetas[10])
            sim.resetJointState(robotId, joint_id_dict['joint12'], targetValue=thetas[11])

            sim.resetJointState(robotId, joint_id_dict['finger1_joint'], targetValue=thetas[12])
            sim.resetJointState(robotId, joint_id_dict['finger2_joint'], targetValue=-thetas[12])

            sim.resetJointState(robotId, joint_id_dict['finger3_joint'], targetValue=thetas[13])
            sim.resetJointState(robotId, joint_id_dict['finger4_joint'], targetValue=-thetas[13])


