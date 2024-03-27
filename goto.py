
import ast
import math
import time

import numpy as np
from robotiq_gripper import RobotiqGripper
import rtde_control
import rtde_receive


def deg_to_rad(l):
    return list(map(math.radians, l))

def goto(rtde_c, rtde_r, gripper, grasp=False, handover=False):

    front = [173.53, -40.66, -144.32, -34.69, 95.20, -3.87]
    mid = [273.93, -46.98, -131.55, -41.77, 61.43, 13.24]
    handoff = [294.44, -95.15, -154.43, 43.15, 53.49, 13.25]

    print('goto 1')
    

    with open('arm_target.txt', 'r') as file:
        content = file.read()

    stage = ast.literal_eval(content.splitlines()[0])
    final = ast.literal_eval(content.splitlines()[1])

    # Open gripper
    gripper.move(gripper.get_open_position(), 64, 1)
    
    print('goto 2')
    rtde_c.moveL(stage, 0.3, 0.3, asynchronous=False)

    print('goto 3')
    rtde_c.moveL(final, 0.3, 0.3, asynchronous=False) 
    
    # Close gripper
    if grasp:
        final_pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), 64, 1)

        if status == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT:
            print("Gripper grasped an object")
            # final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 64, 1)

        else:
            final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 64, 1)
            print('Gripper felt nothing')

    print('goto 4')
    rtde_c.moveL(stage, 0.3, 0.3, asynchronous=False) 

    # Move back out to front
    print('goto 5')
    rtde_c.moveJ(deg_to_rad(mid), 0.7, 0.3, asynchronous=False)

    if handover:
        print('goto 6')
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
    ip_address='192.168.1.123'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = RobotiqGripper()
    gripper.connect(ip_address, 63352)
    gripper.activate(auto_calibrate=False)

    
    goto(rtde_c, rtde_r, gripper, grasp=True, handover=True)