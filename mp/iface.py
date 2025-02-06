#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np

@dataclass
class Command:
    target_q: np.ndarray = None
    left_gripper_open: bool = None
    right_gripper_open: bool = None


