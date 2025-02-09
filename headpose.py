

import sys

import cv2

from gaze_utils.realsense import RSCapture, WebCamCapture
from ptgaze.demo import Demo, init_gaze_track
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

np.set_printoptions(suppress=True)

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
    interpolated_img = img2 * (1 - gradient) + img1 * gradient

    return interpolated_img


tracker = init_gaze_track()

heatmap = np.zeros((720, 1280), dtype=float)
white = np.full((720, 1280, 1), 255, dtype=float)
big_gaussian = np.zeros((720 * 2, 1280 * 2), dtype=float)
draw_gaussian(big_gaussian, 1280, 720, heatmap.shape[1] / 15 / 1.5)

sx = [0] * 9
sy = [0] * 9

def infer(display_image, viz=True): 
    global heatmap, ps

    frame = next(tracker.cap)
           
    vectors = tracker._process_image(frame)

    for vector in vectors:
        vector_x, vector_y, vector_z = vector

        # Time series stuff
        sx.append(vector_x)
        sy.append(vector_y)

        if len(sx) > 9:
            del sx[0]
        if len(sy) > 9:
            del sy[0]

        def ema(series, alpha):
            # Alpha is new thing weight
            p = series[0]
            for thing in series[1:]:
                p = (1 - alpha) * p + alpha * thing
            return p
        
        smooth_x = -ema(sx, 0.3)
        smooth_y = -ema(sy, 0.3)

        smooth_x = smooth_x * 2 - 0.2
        smooth_y = smooth_y * 6 + 0.5
        
        # plt.clf()
        # plt.plot(smooth_x, smooth_y, marker='o', markersize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.pause(0.001)
        # print(smooth_x, smooth_y)

        

        # heatmap = draw_gaussian(heatmap, 
        #                         heatmap.shape[1] * smooth_x, 
        #                         heatmap.shape[0] * smooth_y, 
        #                         heatmap.shape[1] / 15 / 1.5, 1)
        smooth_x = np.clip(smooth_x, 0, 1)
        smooth_y = np.clip(smooth_y, 0, 1)
        gaus_offsetx = heatmap.shape[1] * (1 - smooth_x)
        gaus_offsety = heatmap.shape[0] * (smooth_y)
        heatmap = big_gaussian[int(gaus_offsety) : int(gaus_offsety + heatmap.shape[0]), 
                               int(gaus_offsetx) : int(gaus_offsetx + heatmap.shape[1])]
    
    if viz:
        cv2.namedWindow('heatmap', cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('heatmap',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('heatmap', interpolate_image(white, display_image, heatmap).astype(np.uint8))

        cv2.imshow('frame', tracker.visualizer.image)
        cv2.waitKey(1)

            
    return heatmap

if __name__ == "__main__":
    image = np.asarray(Image.open('image.png'))
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))

    start_time = time.time()
    steps = 0
    while True:
        try:
            heatmap = infer(image, viz=True)
            
            now = time.time()
            # print('fps:', 1 / (now - start_time))
            start_time = now

            steps += 1

            if steps > 300:
                torch.save(heatmap, 'heatmap.pth')
                break
        except KeyboardInterrupt:
            pass