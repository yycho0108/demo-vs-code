from pybullet_utils import bullet_client
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
import pybullet_data
import numpy as np
import pybullet
import time
import numpy

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


class EnvBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot_id = None
        self.obs_ids = {}
        self.obj_ids = {}
        self._load_plane_and_robot(cfg)
        self._set_render_cfg(cfg["render"])

    def _load_plane_and_robot(self, cfg):
        self.sim = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        self.sim.setGravity(0, 0, -9.81)

        # Set dt
        self.sim_hz = cfg["sim_hz"]
        self.ctrl_step = cfg["sim_hz"] // cfg["control_hz"]
        self.dt = 1 / self.sim_hz
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
        return np.array(get_joint_positions(self.robot_id, self.robot.arm_joint_indices))
    
    def set_joint_positions(self, q):
        assert len(q) == self.robot.num_arm_joints, f"Given q {q} is not matched to num_joints {self.robot.num_arm_joints}"
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q)
    
    def set_single_arm_joint_positions(self, q, left_or_right="left"):
        assert left_or_right in ["left", "right"], "left_or_right should be either 'left' or 'right'"
        assert len(q) == self.robot.num_arm_joints//2, f"Given q {q} is not matched to num_joints {self.robot.num_arm_joints//2}"
        
        q_all = self.get_joint_positions()
        if left_or_right == "left":
            q_all[:6] = q
        elif left_or_right == "right":
            q_all[6:] = q
        
        self.set_joint_positions(q_all)
    
    def get_ee_pose(self):
        left_ee_pos, left_ee_quat = get_link_pose(self.robot_id, self.robot.left_arm_joint_indices[-1])
        right_ee_pos, right_ee_quat = get_link_pose(self.robot_id, self.robot.right_arm_joint_indices[-1])
        return np.array([list(left_ee_pos + left_ee_quat), list(right_ee_pos + right_ee_quat)])
    
    def get_gripper_joint_position(self):
        left_finger_pos = np.array(get_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices))
        right_finger_pos = np.array(get_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices))
        return left_finger_pos, right_finger_pos

    def get_gripper_opened(self):
        left_finger_pos, right_finger_pos = self.get_gripper_joint_position()
        is_left_opened = np.diff(left_finger_pos) > (self.robot.left_gripper_joint_limits[1][1] - self.robot.left_gripper_joint_limits[0][0])/2
        is_right_opened = np.diff(right_finger_pos) > (self.robot.right_gripper_joint_limits[1][1] - self.robot.right_gripper_joint_limits[0][0])/2

        return np.concatenate([is_left_opened, is_right_opened])
    
    def set_gripper_open(self, left_or_right="left"):
        if left_or_right == "left":
            set_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices, self.robot.left_gripper_open_pos)
        elif left_or_right == "right":
            set_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices, self.robot.right_gripper_open_pos)
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")
        
    def set_gripper_close(self, left_or_right="left"):
        if left_or_right == "left":
            set_joint_positions(self.robot_id, self.robot.left_gripper_joint_indices, self.robot.left_gripper_close_pos)
        elif left_or_right == "right":
            set_joint_positions(self.robot_id, self.robot.right_gripper_joint_indices, self.robot.right_gripper_close_pos)
        else:
            raise ValueError("left_or_right should be either 'left' or 'right'")
    
    def get_object_poses(self):
        pose_dict = {}
        for obj_name, obj_id in self.obj_ids.items():
            pos, quat = get_pose(obj_id)
            pose_dict[obj_name] = np.array(list(pos) + list(quat))
        return pose_dict
    
    def check_collision(self, q):
        q_orig = get_joint_positions(self.robot_id, self.robot.arm_joint_indices)
        is_collision = get_collision_fn(
            body=self.robot_id,
            joints=self.robot.arm_joint_indices,
            obstacles=self.obs_ids.values(),
            cache=True
        )(q)
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q_orig)
        return is_collision
    
    def solve_tool_ik(self, ee, left_or_right="left", max_attempts=5, check_collision=True):
        assert left_or_right in ["left", "right"], "left_or_right should be either 'left' or 'right'"
        q = self.get_joint_positions()
        q_ik = self.robot.ik(ee, left_or_right, max_attempts=max_attempts)
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, q)
        
        if q_ik is None:
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
            return None

        return q_ik

    def _get_val_fn(self):
        val_fn = get_validate_fn(
            body=self.robot_id,
            joints=self.robot.arm_joint_indices,
            obstacles=self.obs_ids.values(),
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
            positionGains=[0.1]*self.robot.num_arm_joints
        )

    def _set_gripper_command(self, command):
        if command.left_gripper_open is not None:
            if command.left_gripper_open:
                target_left_gripper_vel = self.robot.left_gripper_open_pos
            else:
                target_left_gripper_vel = self.robot.left_gripper_close_pos

            self.sim.setJointMotorControlArray(
                self.robot_id, 
                self.robot.left_gripper_joint_indices, 
                self.sim.POSITION_CONTROL,
                targetPositions=target_left_gripper_vel,
                positionGains=[0.02]*2,
            )

        if command.right_gripper_open is not None:
            if command.right_gripper_open:
                target_right_gripper_vel = self.robot.right_gripper_open_pos
            else:
                target_right_gripper_vel = self.robot.right_gripper_close_pos

            self.sim.setJointMotorControlArray(
                self.robot_id, 
                self.robot.right_gripper_joint_indices, 
                self.sim.POSITION_CONTROL,
                targetPositions=target_right_gripper_vel,
                positionGains=[0.02]*2,
            )

    def execute_command(self, command, render=False):
        imgs = []
        for i in range(len(command)):
            for _ in range(self.ctrl_step):
                self._set_target_joint_position(command[i])
                self._set_gripper_command(command[i])
                self.sim.stepSimulation()
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
        cube_size = [0.02, 0.02, 0.1]
        card_z = table_pos[2] + table_shape[2]/2 + cube_size[2]/2
        self.cube_pos = [table_pos[0], 0.0, card_z]
        self.cube_quat = [0.0, 0.0, 0.0, 1.0]
        cube_id = create_box(*cube_size, mass=0.1, color=BLUE)
        set_pose(cube_id,(self.cube_pos, self.cube_quat))
        self.obj_ids["cube"] = cube_id

    def reset(self):
        set_joint_positions(self.robot_id, self.robot.arm_joint_indices, self.cfg["q_init"])
        set_pose(self.obj_ids["cube"], (self.cube_pos, self.cube_quat))

    
if __name__ == "__main__":
    sim = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Plane
    sim.loadURDF("plane.urdf", [0,0,0])
    # tableUid = sim.loadURDF("table/table.urdf", basePosition = [0, -0.65, -0.12])
    # URDF_PATH = 'RobotArmV3URDF/urdf/RobotArmV3URDF.urdf'
    # PYBULLET_URDF_PATH = 'RobotArmV3URDF/urdf/RobotArmV3URDF.urdf'
    URDF_PATH = 'assets/RobotBimanualV4/urdf/RobotBimanualV4_gripper.urdf'
    PYBULLET_URDF_PATH = 'assets/RobotBimanualV4/urdf/RobotBimanualV4_gripper.urdf'

    L_JOINT_LIMIT_MIN = np.pi/180.*np.array([-90.,  -90., -180.,  -45., -210., -125.]) 
    L_JOINT_LIMIT_MAX = np.pi/180.*np.array([50.,  80., 80.,  75., 210. , 125.])
    R_JOINT_LIMIT_MIN = np.pi/180.*np.array([-50.,  -80., -80.,  -75., -210., -125.]) 
    R_JOINT_LIMIT_MAX = np.pi/180.*np.array([90.,  90., 180.,  45., 210. , 125.])

    # Panda
    startPos = [0, 0, 0]
    startOrientation = sim.getQuaternionFromEuler([0, 0, 0])
    robotId = sim.loadURDF(PYBULLET_URDF_PATH, startPos, startOrientation, useFixedBase=1)

    numJoints = sim.getNumJoints(robotId)
    link_id_dict = dict()
    joint_id_dict = dict()
    for _id in range(numJoints):
        joint_info = sim.getJointInfo(robotId, _id)
        link_name = joint_info[12].decode('UTF-8')
        link_id_dict[link_name] = _id
        joint_name = joint_info[1].decode('UTF-8')
        joint_id_dict[joint_name] = _id
        print(link_name, joint_name, _id)
    
    # sim.changeVisualShape(robotId, link_id_dict['link1'], rgbaColor=[1., 1., 1., 1.])
    # sim.changeVisualShape(robotId, link_id_dict['link2'], rgbaColor=[0., 0., 0., 1.])
    # sim.changeVisualShape(robotId, link_id_dict['link3'], rgbaColor=[0., 0., 1., 1.])
    # sim.changeVisualShape(robotId, link_id_dict['link4'], rgbaColor=[0., 1., 1., 1.])
    # sim.changeVisualShape(robotId, link_id_dict['link5'], rgbaColor=[1., 0., 1., 1.])
    # sim.changeVisualShape(robotId, link_id_dict['link6'], rgbaColor=[1., 1., 0., 1.])


    # link1 joint1 0
    # link2 joint2 1
    # link3 joint3 2
    # link4 joint4 3
    # link5 joint5 4
    # link6 joint6 5

    end_effector_name = 'link6'
    end_effector_name_pybullet = "link6" # link6
    # sim.getLinkState(robotId, link_id_dict[end_effector])[0]

    control_dt = 1./100
    # Control Frequency
    sim.setTimestep = control_dt

    
    debugparams = []
    MODE = 'pos' # 'jp' or 'pos' or 'sinusoidal' or 'inv_dyn'
    init_EE_pos = sim.getLinkState(robotId, link_id_dict[end_effector_name_pybullet])[0]
    
    # ee JP control
    if MODE == 'jp' or MODE == 'inv_dyn':
        debugparams.append(sim.addUserDebugParameter("end-effector X",-0.3,0.3))
        debugparams.append(sim.addUserDebugParameter("end-effector Y",-0.3,0.3))
        debugparams.append(sim.addUserDebugParameter("end-effector Z",-0.3,0.3))

        for i in range(numJoints):
            sim.setJointMotorControl2(robotId, i, sim.VELOCITY_CONTROL, force=0.01)

    elif MODE == 'pos':
        for i in range(6):
            debugparams.append(sim.addUserDebugParameter(f"theta_{i+1}",L_JOINT_LIMIT_MIN[i],L_JOINT_LIMIT_MAX[i],0))
        
        for i in range(6, 12):
            debugparams.append(sim.addUserDebugParameter(f"theta_{i+1}",R_JOINT_LIMIT_MIN[i-6],R_JOINT_LIMIT_MAX[i-6],0))
        
        debugparams.append(sim.addUserDebugParameter(f"left_finger", -45.*np.pi/180., 0, -45.*np.pi/180.))
        debugparams.append(sim.addUserDebugParameter(f"right_finger", -45.*np.pi/180., 0, -45.*np.pi/180.))
            
    
    task = "pen_holder"
    
    if task == "culling":
        table_height = 0.4
        tableShape = (1.0, 0.3, table_height/2)
        tablePosition = (1.1, 0.3, table_height/2)
        boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        tableVisualShapeId = sim.createVisualShape(
            shapeType=sim.GEOM_BOX,
            halfExtents=tableShape,
            rgbaColor=boxColor
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=sim.GEOM_BOX, 
            halfExtents=tableShape
        )
        tableId = sim.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        meshScale = [1.5, 1.5, 1.5]
        #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = sim.createVisualShape(shapeType=sim.GEOM_MESH,
                                            fileName="assets/meshes/culling.stl",
                                            rgbaColor=[1, 0, 0, 1],
                                            specularColor=[0.4, .4, 0],
                                            meshScale=meshScale)
        collisionShapeId = sim.createCollisionShape(shapeType=sim.GEOM_MESH,
                                                fileName="assets/meshes/culling.stl",
                                                meshScale=meshScale)


        cullingId = sim.createMultiBody(baseMass=1,
                        baseInertialFramePosition=[0, 0, 0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.3, 0.3, table_height],
                        baseOrientation = [0, 0, 0.7071068, 0.7071068],
                        useMaximalCoordinates=True)
        

    if task == "pen_holder":
        table_height = 0.4
        tableShape = (0.2, 0.42, table_height/2)
        tablePosition = (0.3, 0.0, table_height/2)
        boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        tableVisualShapeId = sim.createVisualShape(
            shapeType=sim.GEOM_BOX,
            halfExtents=tableShape,
            rgbaColor=boxColor
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=sim.GEOM_BOX, 
            halfExtents=tableShape
        )
        tableId = sim.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        meshScale = [1., 1., 1.]
        #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = sim.createVisualShape(shapeType=sim.GEOM_MESH,
                                            fileName="assets/meshes/penholder.stl",
                                            rgbaColor=[0, 1, 0, 1],
                                            specularColor=[0.4, .4, 0],
                                            meshScale=meshScale)
        collisionShapeId = sim.createCollisionShape(shapeType=sim.GEOM_MESH,
                                                fileName="assets/meshes/penholder.stl",
                                                meshScale=meshScale)


        holderId = sim.createMultiBody(baseMass=1,
                        baseInertialFramePosition=[0, 0, 0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.3, 0., table_height],
                        useMaximalCoordinates=True)
        
        # meshScale = [1., 1., 1.]
        # #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        # visualShapeId = sim.createVisualShape(shapeType=sim.GEOM_MESH,
        #                                     fileName="pen.stl",
        #                                     rgbaColor=[0, 1, 0, 1],
        #                                     specularColor=[0.4, .4, 0],
        #                                     meshScale=meshScale)
        # collisionShapeId = sim.createCollisionShape(shapeType=sim.GEOM_MESH,
        #                                         fileName="pen.stl",
        #                                         meshScale=meshScale)


        # penId = sim.createMultiBody(baseMass=1,
        #                 baseInertialFramePosition=[0, 0, 0],
        #                 baseCollisionShapeIndex=collisionShapeId,
        #                 baseVisualShapeIndex=visualShapeId,
        #                 basePosition=[0.3, 0., table_height],
        #                 useMaximalCoordinates=True)
    
    



    sim.setRealTimeSimulation(False)
    sim.setGravity(0, 0, -9.81)
    # sim.setGravity(0, 0, 0)

    # timeStepId = sim.addUserDebugParameter("timeStep", 0.001, 0.1, 0.01)

    Start_Simul = True
    while Start_Simul:
        sim.stepSimulation()

        # timeStep = sim.readUserDebugParameter(timeStepId)
        sim.setTimeStep(control_dt)
        # time.sleep(control_dt)
        thetas = []
        for param in debugparams:
           thetas.append(sim.readUserDebugParameter(param))
        
        if MODE == 'pos':
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

            # G_torque = getGravityCompensation(robotId)
            # print(f'Gravity Compensation Torque = {G_torque}')

            cur_joint_pos = []
            # for i in range(3):
            #     link_name = f'link{i+1}'
            #     joint_pos = sim.getJointState(robotId, link_id_dict[link_name])[0]
            #     cur_joint_pos.append(joint_pos)
            # cur_joint_pos = np.array(cur_joint_pos)
            # print(f'Gravity Compensation Torque = {getGravityCompensation(robotId)}')
            # print(cur_joint_pos)


        # time.sleep(control_dt)
        keys = sim.getKeyboardEvents()

        for k,v in keys.items():
            if k==113 and (v & sim.KEY_WAS_TRIGGERED):
                Start_Simul = False
                sim.disconnect()


