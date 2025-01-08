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


