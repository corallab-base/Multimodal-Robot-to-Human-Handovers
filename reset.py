# Imports
import math
import rtde_control
import rtde_receive
from gaze_utils.constants import front, mid, handoff

'''
Sometimes we'd like to bring the arm to a certain position without having to use the tablet
This script lets us do this without leaving remote mode
'''

def deg_to_rad(l):
    return list(map(math.radians, l))

def goto(rtde_c, rtde_r):
    # Move back out to front
    rtde_c.moveJ(deg_to_rad(front), 1, 1, asynchronous=False)
    
if __name__ == '__main__':
    ip_address='192.168.1.125'

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
    goto(rtde_c, rtde_r)










def compute_iou(box1, box2):
    # Unpack the coordinates of the two boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    # Calculate the area of intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate the area of both bounding boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    
    # Calculate the area of union
    union_area = area_box1 + area_box2 - inter_area
    
    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou