from dataclasses import dataclass
import numpy as np

from pybullet_planning.pybullet_tools.utils import *

from ompl import base as ob
from ompl import geometric as og


@dataclass
class Command:
    target_q: np.ndarray = None
    left_gripper_open: bool = None
    right_gripper_open: bool = None


def ompl_state_to_np(state, num_joints):
    return np.array([state[i] for i in range(num_joints)])


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class MotionPlanner:
    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
        self._get_robot_info(env.robot)
        self._init_ompl()


    def _get_robot_info(self, robot):
        self.robot_id = robot.robot_id

        # Joint indices without gripper
        self.num_joints = robot.num_arm_joints
        self.joint_indices = robot.arm_joint_indices

        # Joint limits
        self.joint_limits = robot.arm_joint_limits


    def _init_ompl(self):
        # Set up configuration space
        self.space = PbStateSpace(self.num_joints)

        # Set joint limit
        limits = ob.RealVectorBounds(self.num_joints)
        for i in range(self.num_joints):
            limits.setLow(i, self.joint_limits[0][i])
            limits.setHigh(i, self.joint_limits[1][i])
        self.space.setBounds(limits)

        # Return False when collision.
        self.val_fn = self.env._get_val_fn()

    def get_joint_command(
            self, 
            q_goal=None, 
            left_tool_goal=None, 
            right_tool_goal=None, 
            q_start=None, 
            open_gripper=None,
            timeout=1.0, 
            simplify=True, 
            interpolation_res=0.01, 
            attached_obj_name=None,
        ):
        q_orig = self.env.get_joint_positions()

        if open_gripper is not None:
            gripper_command = [open_gripper, open_gripper]
        else:
            gripper_command = self.env.get_gripper_opened()

        if q_start is None:
            q_start = self.env.get_joint_positions()
        else:
            assert len(q_start) == self.num_joints, f"Given q_start {q_start} is not matched to num_joints {self.num_joints}"
        
        assert q_goal is not None or left_tool_goal is not None or right_tool_goal is not None, "Either q_goal or tool_goal should be given."
        if q_goal is not None:
            assert len(q_goal) == self.num_joints, f"Given q_goal {q_goal} is not matched to num_joints {self.num_joints}"
        else:
            q_goal = q_start.copy()
            if left_tool_goal is not None:
                assert len(left_tool_goal) == 7, f"Given tool_goal {left_tool_goal} is not matched to 7"
                left_q_goal = self.env.solve_tool_ik(left_tool_goal, "left")
                if left_q_goal is None:
                    print("No IK solution found for left_tool_goal!!")
                    return []
                q_goal[:self.num_joints//2] = left_q_goal
            
            if right_tool_goal is not None:
                right_q_goal = self.env.solve_tool_ik(right_tool_goal, "right")
                if right_q_goal is None:
                    print("No IK solution found for right_tool_goal!!")
                    return []
                q_goal[self.num_joints//2:] = right_q_goal

        ss = og.SimpleSetup(self.space)

        # Set validate function (Reverse of collision function)
        if attached_obj_name is None:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.val_fn))
        else:
            attach_val_fn = self.env._get_attached_collision_fn(attached_obj_name)
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(attach_val_fn))

        # Set motion planning algorithm
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.0025)
        ss.setPlanner(og.RRTConnect(si))

        # Set start and goal state
        q0 = ob.State(self.space)
        qG = ob.State(self.space)
        for i in range(self.num_joints):
            q0[i] = q_start[i]
            qG[i] = q_goal[i]
        ss.setStartAndGoalStates(q0, qG)

        # Solve
        if not ss.solve(timeout):
            # No solution found. Return empty list
            print("No motion planning solution found!!")
            return []

        # Simplify solution such as shortcut, reduce vertices, smoothBSpline,
        if simplify:
            ss.simplifySolution()

        # Get solution path
        traj = ss.getSolutionPath()

        # Interpolate trajectory
        traj.interpolate(int(traj.length()/interpolation_res))

        self.env.set_joint_positions(q_orig)

        commands = []
        for state in traj.getStates():
            commands.append(
                Command(
                    target_q=ompl_state_to_np(state, self.num_joints),
                    left_gripper_open = gripper_command[0],
                    right_gripper_open = gripper_command[1]
                )
            )

        return commands
