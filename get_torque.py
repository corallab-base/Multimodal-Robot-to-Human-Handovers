
import ast
import math

from matplotlib import pyplot as plt
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from robotiq_gripper import RobotiqGripper
import rtde_control
import rtde_receive

def torque(rtde_c, rtde_r, gripper):
    hist = []
    while True:
        f = rtde_r.getActualTCPForce()
        m = np.linalg.norm(f[3:])
        print(m)
        hist.append(f[3:])

        for i, thing in enumerate(zip(*hist)):
            plt.plot(range(len(thing)), hist, label=f'{i}')
        plt.legend()
        plt.pause(0.001)

if __name__ == '__main__':
    ip_address='192.168.1.123'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = RobotiqGripper()
    gripper.connect(ip_address, 63352)
    gripper.activate(auto_calibrate=False)

    
    torque(rtde_c, rtde_r, gripper)