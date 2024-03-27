
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

    front = [164.88, -46.17, -124.62, -53.24, 101.27, -10.25]
    mid = [273.93, -46.98, -131.55, -41.77, 61.43, 13.24]
    handoff = [294.44, -95.15, -154.43, 43.15, 53.49, 13.25]

    # Move back out to front
    rtde_c.moveJ(deg_to_rad(handoff), 1.5, 0.9, asynchronous=False)
    
if __name__ == '__main__':
    ip_address='192.168.1.123'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
    goto(rtde_c, rtde_r)