
from matplotlib import pyplot as plt
import numpy as np


def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """
    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb

# Util function to display a point cloud
def view_pc(pc, col=None, show=False, title=None):
    # Setting up the figure
    fig = plt.figure()

    if title is not None:
        plt.title(title)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    if col is None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=col)

    # Set the axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

    if show: plt.show()

    
