import os
import argparse
import sys
import torch
sys.path.append('../')

import cv2
import matplotlib.pyplot as plt
from random_poses import random_poses
# from mover import init as initArm

def count_npz_files(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    
    # Filter files to include only those with the .npz extension
    npz_files = [file for file in files if file.endswith('.pth')]
    
    # Count the number of .npz files
    num_npz_files = len(npz_files)
    
    return num_npz_files

def viz_pos(positions):
    # Visualizing the generated scenes
    for name, (x, y) in positions:
        plt.scatter(x, y, s=2000) # Plot each point
        plt.text(x, y, name, rotation=30, fontsize=10) # Annotate each point with its name

    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().axvline(-0.5)
    plt.gca().axvline(0, linestyle='--')
    plt.gca().axvline(0.5)
    plt.gca().axhline(0)
    plt.gca().axhline(0.5, linestyle='--')
    plt.gca().axhline(1.0)
    plt.gcf().set_size_inches(9, 9)
    plt.show()

if __name__ == "__main__":
    print('Capture images from the wrist-mounted camera and save to datasets/scenes/raw_N.pth.'
          'Place everything in front of the midline of the table so that the arm can reach it comfortably'
          'Make sure all objects are in frame via the preview image.')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=int, default=None, help='ID of the scene. Leave empty to infer')
    parser.add_argument('--save_dir', type=str, default='./scenes', help='Directory to save the scenes')
    
    args = parser.parse_args()

    if args.id is None:
        args.id = count_npz_files(args.save_dir) + 1
    
    positions, queries, object_parts  = random_poses()

    # front_cam_pose, front_rgb, front_depth = initArm(True)

    while True:
        viz_pos(positions)
        # front_cam_pose, front_rgb, front_depth = initArm(True)

        # image = cv2.cvtColor(front_rgb, cv2.COLOR_BGR2RGB)
        # plt.imshow(image, cmap='viridis')
        # plt.axis('off')  # Turn off axis
        # plt.show()
        # inp = input("Redo (y or n):")
        # if inp == 'n':
            #  break

    # data_to_save = {
    #     'front_cam_pose': front_cam_pose,
    #     'front_rgb': front_rgb,
    #     'front_depth': front_depth,
    #     'object_positions': positions,
    #     'object_parts': object_parts,
    #     'queries': queries
    # }

    # torch.save(data_to_save, f'./scenes/scene_{args.id}.pth')


def get_scene(scene_num):
    data = torch.load(f'./scenes/scene_{scene_num}.pth')
    print('object_parts', data['object_parts'], 'queries', data['queries'])