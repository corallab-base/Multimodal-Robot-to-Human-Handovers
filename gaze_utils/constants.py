
import numpy as np

# At 1280x720
# Empirically determined
d415_intrinsics = np.array([[952.828 * 0.94, 0, 646.699], [0, 952.828 * 0.94,     342.637 ], [0, 0,         1.  ]] )


d455_intrinsics = np.array([[911.445649104, 0, 641.169], [0, 891.51236121, 352.77], [0, 0, 1]])

# In camera coordinates: [(-)left-to-right(+), (-)up-to-down(+), (-)back-to-front(+), rotvec ]
cam_tool_offset = [0.01, -0.075, 0.05, 0, 0, 0]

front = [0.21, -89.10, 157.20, -231.12, -90.52, 0.34]
mid = [273.93, -46.98, -131.55, -41.77, 61.43, 13.24]
handoff = [294.44, -95.15, -154.43, 43.15, 53.49, 13.25]