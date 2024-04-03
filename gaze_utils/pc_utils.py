import open3d as o3d

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

    indices = np.random.choice(pc.shape[0], size=50000, replace=False)
    pc = pc[indices]
    col = col[indices]
    
    # Plot the point cloud data
    if col is None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=col/256)

    # Set the axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

    if show: plt.show()

    
def view_pc_o3d(pc, col=None, show=False, title=None):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(pc)

    # If colors are provided, apply them
    if col is not None:
        # Ensure color is in the format Open3D expects: [0, 1]
        # Assuming 'col' is provided as an Nx3 matrix with values in [0, 255]
        col = np.asarray(col) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(col)

    # Visualization
    # Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.show_coordinate_frame = True

    view_ctl = vis.get_view_control()

    # Set front, lookat, up, and zoom parameters to adjust the camera
    # These values might need to be adjusted depending on the point cloud
    front = [-1, 0, 0]  # Direction where the camera is facing
    lookat = [0, 0, 0]  # The point at which the camera is looking
    up = [0, 1, 0]      # Setting Z as the up direction
    zoom = 0.5

    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

    # Optionally return the Open3D point cloud object for further processing
    return pcd