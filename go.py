
import math
import rtde_control
import rtde_receive

ip_address='192.168.1.123'

rtde_c = rtde_control.RTDEControlInterface(ip_address)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

front = [164.88, -46.17, -124.62, -53.24, 101.27, -10.25]

dest = \
[-0.7314032341476323, 0.03497839184122027, 0.34853320072375693, 1.2801981693573552, -2.8004493016773115, -0.2901314802485546]

dest2 = \
[-0.7496653596694848, 0.057418752466677485, 0.20134989668955763, 1.2801981693573552, -2.8004493016773115, -0.2901314802485546]

def deg_to_rad(l):
    return list(map(math.radians, l))

rtde_c.moveL(dest, 0.3, 0.3, asynchronous=False)

rtde_c.moveL(dest2, 0.3, 0.3, asynchronous=False)

