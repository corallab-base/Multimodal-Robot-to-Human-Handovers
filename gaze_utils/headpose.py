

import sys
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

# from img2pose.utils.renderer import Renderer
from img2pose.utils.image_operations import expand_bbox_rectangle
from img2pose.utils.pose_operations import get_pose
from img2pose.img2pose import img2poseModel
from img2pose.model_loader import load_model

np.set_printoptions(suppress=True)

# renderer = Renderer(
#     vertices_path="img2pose/pose_references/vertices_trans.npy", 
#     triangles_path="img2pose/pose_references/triangles.npy"
# )

def render_plot(img, poses, bboxes):
    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    
    trans_vertices = renderer.transform_vertices(img, poses)
    img = renderer.render(img, trans_vertices, alpha=1)    

    plt.figure(figsize=(8, 8))     
    
    for bbox in bboxes:
        plt.gca().add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=3,edgecolor='b',facecolor='none'))            
    
    plt.imshow(img)        
    plt.show()

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
    del img2pose_model


def infer(img, threshold = 0.9, viz=False): 
    h, w = img.shape[0:2]
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
            
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

    if viz:
        render_plot(img.copy(), poses, bboxes)


    return poses, bboxes