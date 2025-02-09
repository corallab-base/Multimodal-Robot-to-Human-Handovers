
import math
import numpy as np

# At 1280x720
# Empirically determined
# d415_intrinsics = np.array([[952.828 * 1.03, 0, 646.699], [0, 952.828 * 0.94,     342.637 ], [0, 0,         1.  ]] )
d415_intrinsics = np.array([[952.828 * 0.94, 0, 646.699 * 0.94], [0, 952.828,     342.637 ], [0, 0,         1.  ]] )


d455_intrinsics = np.array([[911.445649104, 0, 641.169], [0, 891.51236121, 352.77], [0, 0, 1]])

# In camera coordinates: [(-)left-to-right(+), (-)up-to-down(+), (-)back-to-front(+), rotvec ]
cam_tool_offset = [0.01, -0.075, 0.05, 0, 0, 0]

front = [90, -150, 135, 285, -45, -90]
mid = [49.62, -109.03, 117.40, 264.70, -86.09, -130.33]
handoff = [-42.83, -94.89, 136.51, 195.12, -127.08, -325.95]


def camera_tweaks(roll, pitch, yaw):
   '''
   Given the orientation (in an ideal world with a mount that isn't made of plastic) of the camera,
   convert it to what it actually is (extrinsic euler)
   '''

   # In the following I determine the error in the camera mount experimentally
   # Sorry, the roll/pitch/yaw names don't actually correspond
   # The directional comments correspond to how the tweak affects the pointcloud,
   # not the rotation itself 

   return [roll + math.radians(-5), # Positive is up (yaw)
          pitch + math.radians(-3), # Positive is clockwise spin
          yaw + math.radians(3)] # Positive is right

