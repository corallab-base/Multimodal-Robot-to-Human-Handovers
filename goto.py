
import ast
import math
import time

import numpy as np
from robotiq_gripper import RobotiqGripper
import rtde_control
import rtde_receive
from gaze_utils.constants import front, mid, handoff

def deg_to_rad(l):
    return list(map(math.radians, l))

def goto(rtde_c, rtde_r, gripper, grasp=True, handover=False):
    with open('arm_target.txt', 'r') as file:
        content = file.read()

    stage = ast.literal_eval(content.splitlines()[0])
    final = ast.literal_eval(content.splitlines()[1])

    # Open gripper - going to the object
    print("Going to the object...")
    gripper.move(gripper.get_open_position(), 64, 1)
    rtde_c.moveL(stage, 0.3, 0.3, asynchronous=False)
    rtde_c.moveL(final, 0.3, 0.3, asynchronous=False) 
    
    # Close gripper
    if grasp:
        final_pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), 64, 1)
        if status == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT:
            print("Gripper grasped an object")
        else:
            print('Gripper felt nothing')


    print("Handing over the object...")
    stage[2] += 0.02
    rtde_c.moveL(stage, 0.4, 0.4, asynchronous=False) 
    rtde_c.moveJ(deg_to_rad(mid), 0.7, 0.3, asynchronous=False)


    if handover:
        rtde_c.moveJ(deg_to_rad(handoff), 0.7, 0.3, asynchronous=False)
        
        # Measure regular torque for 0.5s
        torque_sum = 0
        torque_num = 0
        start = time.time()
        while time.time() - start < 0.5:
            f = rtde_r.getActualTCPForce()
            m = np.linalg.norm(f[3:])
            torque_sum += m
            torque_num += 1
        
        torque_baseline = torque_sum / torque_num

        # Let go when torque deviates from baseline
        while True:
            f = rtde_r.getActualTCPForce()
            m = np.linalg.norm(f[3:])
            # print('baseline', torque_baseline, m)
            if abs(torque_baseline - m) > 0.3:
                gripper.move(gripper.get_open_position(), 64, 1)
                break
    
if __name__ == '__main__':
    ip_address='192.168.1.125'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = RobotiqGripper()
    gripper.connect(ip_address, 63352)
    gripper.activate(auto_calibrate=False)

    
    goto(rtde_c, rtde_r, gripper, grasp=True, handover=True)