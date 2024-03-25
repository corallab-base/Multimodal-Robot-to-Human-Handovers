
import ast
import math
import rtde_control
import rtde_receive


def deg_to_rad(l):
    return list(map(math.radians, l))

def goto(rtde_c, rtde_r):

    front = [164.88, -46.17, -124.62, -53.24, 101.27, -10.25]

    print('goto 1')
    

    with open('arm_target.txt', 'r') as file:
        content = file.read()

    stage = ast.literal_eval(content.splitlines()[0])
    final = ast.literal_eval(content.splitlines()[1])

    print('goto 2')
    rtde_c.moveL(stage, 0.3, 0.3, asynchronous=False)

    print('goto 3')
    rtde_c.moveL(final, 0.3, 0.3, asynchronous=False) 
    
    print('goto 4')
    rtde_c.moveL(stage, 0.3, 0.3, asynchronous=False) 

    # Close gripper
    # final_pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), 100, 1)

    # if status == robotiq_gripper.RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT:
    #     print("Gripper grasped an object")
    #     final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 100, 1)

    # else:
    #     final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 100, 1)
    #     print('Gripper felt nothing')

    # Move back out to front
    rtde_c.moveJ(deg_to_rad(front), 1.5, 0.9, asynchronous=False)
    
if __name__ == '__main__':
    ip_address='192.168.1.123'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
    goto(rtde_c, rtde_r)