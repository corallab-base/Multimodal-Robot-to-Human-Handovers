# General Imports
import os
from util import *
import numpy as np
from time import sleep
from constants import *
from gym_config import *
from kinematics import *
from isaacgym import gymapi
import matplotlib.pyplot as plt
from main import execute_RMP_path
from world_builder import create_table, create_target_box
import CoGrasp.CoGrasp.contact_graspnet_util as cgu

import rtde_control
import rtde_receive

def main(ip_address):
    # Setup for real robot
    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    rtde_c.freedriveMode()

if __name__ == '__main__':
    ip_address = '192.168.0.11'
    main(ip_address)
