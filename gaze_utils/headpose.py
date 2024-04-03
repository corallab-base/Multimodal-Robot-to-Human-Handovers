

import sys

import cv2

from gaze_utils.realsense import RSCapture, WebCamCapture
sys.path.append('../../')
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import matplotlib.patches as patches
import matplotlib
matplotlib.use("TkAgg")

from scipy.spatial.transform import Rotation
import pandas as pd
from scipy.spatial import distance
import time
import os
import math
import scipy.io as sio

from img2pose.utils.renderer import Renderer
from img2pose.utils.image_operations import expand_bbox_rectangle
from img2pose.utils.pose_operations import get_pose
from img2pose.img2pose import img2poseModel
from img2pose.model_loader import load_model

np.set_printoptions(suppress=True)

renderer = Renderer(
    vertices_path="img2pose/pose_references/vertices_trans.npy", 
    triangles_path="img2pose/pose_references/triangles.npy"
)

def render_plot(img, poses, bboxes):
    (w, h) = img.shape[0:2]
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    
    print('poses', poses)

    trans_vertices = renderer.transform_vertices(img, poses)

    print('trans_vertices', trans_vertices)

    img = renderer.render(img, trans_vertices, alpha=1)    

    # plt.figure(figsize=(8, 8))  

    # def draw_bb(arr, bounding_box, color):
    #     # xyxy
    #     bounding_box = [int(x) for x in bounding_box]

    #     bounding_box = [bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]]

    #     print(bounding_box)

    #     arr[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    #     arr[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    #     arr[bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    #     arr[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0] + bounding_box[2]] = color
   


    # for bbox in bboxes:
    #     draw_bb(img, bbox, [0, 0, 255])
    #     # xyxy
    #     # plt.gca().add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=3,edgecolor='b',facecolor='none'))            
    
    plt.gca().imshow(img[..., ::-1]) 

def draw_gaussian(image, x, y, sigma=10, amplitude=1.0):
    """
    Draw a Gaussian on a 2D image at a given x, y coordinate.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - x, y: Coordinates where the Gaussian is centered.
    - sigma: Standard deviation of the Gaussian.
    - amplitude: Height of the Gaussian.
    """
    # Create a meshgrid for the coordinates
    rows, cols = image.shape
    [X, Y] = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calculate the Gaussian kernel
    gaussian = amplitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    
    # Add the Gaussian to the image
    image += gaussian
    
    # Optionally, clip values to maintain valid intensities
    image = np.clip(image, 0, 255)
    
    return image


def spherical2cartesial(x): 
    return -np.cos(x[1]) * np.cos(x[0]), np.cos(x[1]) * np.sin(x[0]), np.sin(x[1])

def interpolate_image(img1, img2, gradient):
    """
    Interpolate between two images using a gradient mask.

    Parameters:
    - img1: First image as an NxMx3 numpy array.
    - img2: Second image as an NxMx3 numpy array.
    - gradient: NxM array of blend ratios, where 0 means all of img1 and 1 means all of img2.

    Returns:
    - Interpolated image as an NxMx3 numpy array.
    """
    # Ensure the gradient is in the shape NxMx1 to perform element-wise multiplication correctly
    gradient = gradient[:, :, np.newaxis]

    # Ensure gradient values are within [0, 1]
    gradient = np.clip(gradient, 0, 1)

    # Interpolate between the images
    interpolated_img  = img2 * (1 - gradient) + img1 * gradient

    return interpolated_img

threed_points = np.load('img2pose/pose_references/reference_3d_68_points_trans.npy')

transform = transforms.Compose([transforms.ToTensor()])

DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

POSE_MEAN = "img2pose/models/WIDER_train_pose_mean_v1.npy"
POSE_STDDEV = "img2pose/models/WIDER_train_pose_stddev_v1.npy"
MODEL_PATH = "img2pose/models/img2pose_v1.pth"

pose_mean = np.load(POSE_MEAN)
pose_stddev = np.load(POSE_STDDEV)

img2pose_model = img2poseModel(
    DEPTH, MIN_SIZE, MAX_SIZE, 
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)

load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()

def free_model():
    global img2pose_model

    img2pose_model.fpn_model.cpu()
    del img2pose_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


heatmap = np.zeros((720 // 8, 1280 // 8), dtype=float)
white = np.full((720, 1280, 1), 255, dtype=float)

ps = [(0, 0)] * 9

def infer(img, display_image, threshold = 0.9, viz=True): 
    global img2pose_model, heatmap, ps


    heatmap[...] = 0

    h, w = img.shape[0:2]
            
    res = img2pose_model.predict([transform(img)])[0]

    all_bboxes = res["boxes"].cpu().numpy().astype('float')

    poses = []
    bboxes = []
    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()

            poses.append(pose_pred / np.array((1., 1., 1., 20., 20., 20.)))
            bboxes.append(bbox)


    from mpl_toolkits.mplot3d import Axes3D
    # render_plot(img.copy(), poses, bboxes)    

    if len(poses) > 1:
        print('Warning, more than one face detected')

    for pose in poses:
        # Unpack pose components
        vector_x, vector_y, vector_z, x, y, z = pose
        x, y, z = 0, 0, 0

        rotvec = Rotation.from_rotvec([vector_x, vector_y, vector_z])
        vector_x, vector_y, vector_z = rotvec.as_euler('xyz')

        # ax = plt.gcf().add_subplot(121, projection='3d')
        # ax.clear()
        # ax.set_xlim(-1, 1) 
        # ax.set_ylim(-1, 1) 
        # ax.set_zlim(-1, 1) 
        
        # # Plot the pose as a point
        # ax.scatter(x, y, z, color='blue', s=100, label='Pose')

        # # Plot the ray from the pose
        # # We'll extend the vector for visualization purposes
        # ray_length = 1
        # ax.quiver(x, y, z, vector_x, vector_y, vector_z, length=ray_length, color='red', arrow_length_ratio=0.1, label='Ray')
        
        # Setting labels for axes
        # ax.set_xlabel('Left Right X')
        # ax.set_ylabel('Front Back Y')
        # ax.set_zlabel('Up Down Z')
        
        # Time series stuff
        ps.append((vector_x, vector_y))

        if len(ps) > 9:
            del ps[0]

        xx = [a for a,b in ps]
        yy = [b for a,b in ps]

        def ema(series, alpha):
            # Alpha is new thing weight
            p = series[0]
            for thing in series[1:]:
                p = (1 - alpha) * p + alpha * thing
            return p
        
        smooth_y = ema(xx, 0.6)
        smooth_x = ema(yy, 0.6)

        smooth_y = smooth_y * 1.7 + 0.5
        smooth_x = smooth_x * 2 + 0.5 

        heatmap = draw_gaussian(heatmap, 
                                heatmap.shape[1] * smooth_x, 
                                heatmap.shape[0] * smooth_y, 
                                heatmap.shape[1] / 15 / 1.5, 1)
    
    if viz:
        heatmap2 = cv2.resize(heatmap, (heatmap.shape[1] * 8, heatmap.shape[0] * 8))
        cv2.namedWindow('heatmap', cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('heatmap',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('heatmap', interpolate_image(white, display_image, heatmap2).astype(np.uint8))
        # cv2.resizeWindow('heatmap', heatmap2.shape[1], heatmap2.shape[0])
        cv2.waitKey(1)

    # plt.imshow(heatmap)
    # plt.pause(0.001)
            
    return heatmap

if __name__ == "__main__":
    v_reader = RSCapture(serial_number='123122060050', dont_set_depth_preset=True)

    for frame in v_reader:
        color_image, depth_image, depth_frame, color_frame = frame
        infer(color_image, color_image, viz=True)