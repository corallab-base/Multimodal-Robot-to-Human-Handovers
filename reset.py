
import ast
import math
import rtde_control
import rtde_receive

'''
Sometimes we'd like to bring the arm to a certain position without having to use the tablet
This script lets us do this without leaving remote mode
'''

def deg_to_rad(l):
    return list(map(math.radians, l))

def goto(rtde_c, rtde_r):
    from gaze_utils.constants import front, mid, handoff

    # Move back out to front
    rtde_c.moveJ(deg_to_rad(front), 0.3, 0.3, asynchronous=False)
    
if __name__ == '__main__':
    ip_address='192.168.1.123'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
    goto(rtde_c, rtde_r)