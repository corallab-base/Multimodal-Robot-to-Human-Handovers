import torch
import numpy as np
import cv2
from scipy import spatial
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sympy import Point3D, Plane, Line3D
from fps_limiter import LimitFPS, FPSCounter

random_colors = np.random.uniform(0, 1, (1000, 3))
random_colors = np.clip(random_colors, 0, 200)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

# Inject the function in Axes3D
setattr(Axes3D, 'arrow3D', _arrow3D)

def sum_last_n(list, n):
    start_i = max(0, len(list) - n)
    
    accum = np.zeros((IMAGE_H, IMAGE_W), dtype=float)
    start_i += 1
    while start_i < len(list):
        accum += list[start_i]
        start_i += 1

    return accum

# x = [-0.05, -0.40, -0.40, -0.05]
# y = [-0.20, -0.20, -0.43,-0.43]
# z = [0.05, 0.05, 0.05, 0.05]

# # plane points for sympy
# a1 = Point3D (-0.05, 0.05, -0.20)
# a2 = Point3D (0.40, 0.05, 0.20)
# a3 = Point3D (-0.40, 0.05, -0.43)

# Define the screen location (meters)
# z points positive towards the user
minx = -0.27
maxx = 0.27
miny = -0.10 - 0.27
maxy = 0.2
offsetz = 0.0
real_width = maxx - minx
real_height = maxy - miny

# Plane verticies for pyplot visualizer
x = [maxx, minx, minx, maxx]
y = [maxy, maxy, miny, miny]
z = [offsetz, offsetz, offsetz, offsetz]

# plane points for sympy
a1 = Point3D (maxx, offsetz, maxy)
a2 = Point3D (minx, offsetz, maxy)
a3 = Point3D (minx, offsetz, miny)

plane = Plane(a1, a2, a3)

verts = [list(zip(x,z,y))]


# fig.tight_layout()

fps_limiter = LimitFPS(fps=30)
fps_counter = FPSCounter()

isx = []
isy = []
isz = []

skip_frame = 1

IMAGE_W = 1280 
IMAGE_H = 720
heatmap_buf = []

def gaussian_like(shape, center, stdev):
    X = np.arange(shape[1])
    Y = np.arange(shape[0])
    X, Y = np.meshgrid(X, Y)

    x_var = (X - center[1])**2
    y_var = (Y - center[0])**2
    gaussian = 500.0 / (2. * 3.141592 * stdev**2) * np.exp(-(x_var + y_var) / (2. * stdev**2))
    return gaussian

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

j = 0
def show_heatmap(frame, viz=False):
    global j
    global heatmap_buf
    global IMAGE_W
    global IMAGE_H

    heatmap_viz = np.zeros((IMAGE_H, IMAGE_W), dtype=float)

    if j % skip_frame != 0:
        return

    if viz:
        ax.clear()
        ax.set_xlim(-0.3, 0.3) # x
        ax.set_ylim(0, 1) # z (front-back)
        ax.set_zlim(-0.3, 0.3) # y

        ax.set_title('People pose and gazes')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')

        ax.add_collection3d(Poly3DCollection(verts, facecolors=['none']*5, linewidths=[2]*5, edgecolors=['red']))

    for face_id, (pose, gaze_raw) in [a for a in frame.items() if type(a[1]) != np.ndarray]:
        # print(pose, gaze)

        # if fps_limiter():
        #     print("elapsed seconds: %s" % fps_limiter.get_elapsed_seconds())

        gaze = gaze_raw * 0.5

        print('gaze:', gaze)

        # Something has a bias to the left, so push the gaze vector right
        gaze[0] += math.radians(20)
        # Something has a bias down, so push the gaze vector up
        gaze[2] += math.radians(5)

        # sympy iintersect
        p0 = Point3D(pose[0], pose[2], -pose[1]) 
        # gaze[2] -= 15 
        v0 = (-gaze[0], gaze[2], gaze[1])
        line = Line3D(p0, direction_ratio=v0)
        intr = plane.intersection(line)
        intersection = np.array(intr[0], dtype=float)

        isx.append(intersection[0])
        isy.append(intersection[1])
        isz.append(intersection[2])

        distance = spatial.distance.euclidean((pose[0], pose[2], -pose[1]) , intersection)
        
        if viz:
            ax.scatter(isx, isy, isz, color='red')

            # swap 
            # x -> x
            # y -> z
            # z -> y
            ax.scatter(pose[0], pose[2], -pose[1], color='green')
            ax.quiver(pose[0], pose[2], -pose[1], # flip y
                    -gaze[0], gaze[2], gaze[1],
                    length=0.3, # head size
                    normalize=True,
                    color=random_colors[face_id].tolist())

        screen_coord = ((-intersection[2] - minx) / real_width * IMAGE_H,
                        (intersection[0] - miny) / real_height * IMAGE_W)
        
        gauss_map = gaussian_like((IMAGE_H, IMAGE_W), screen_coord, 4./60 * distance * IMAGE_H / 0.30)

        heatmap_buf.append(gauss_map)

        heatmap_viz = sum_last_n(heatmap_buf, 8)

        print("gauss max", gauss_map.max())

        cv2.circle(heatmap_viz, (int(screen_coord[1]), int(screen_coord[0])), 15, (0), -1)
    
    # if 'frame' in frame:
    #    cv2.imshow('video', cv2.cvtColor(frame['frame'], cv2.COLOR_BGR2RGB))

    # Dot at gaze
    
    if viz:
        plt.figure(12)
        plt.pause(0.001)

    cv2.imshow("Heatmap", heatmap_viz)
    cv2.waitKey(1)
    
    j += 1

def get_heatmap(n):
    global heatmap_buf
    return sum_last_n(heatmap_buf, 8)

if __name__ == "__main__":
    print('Loading data from gaze_tracking_results.pt')
    data = torch.load('gaze_tracking_results.pt')
    for frame in data:
        show_heatmap(frame)

        cv2.waitKey(1)

        
