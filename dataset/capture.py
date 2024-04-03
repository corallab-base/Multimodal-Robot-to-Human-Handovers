import argparse
import os

import torch
from mover import init as initArm

def count_npz_files(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    
    # Filter files to include only those with the .npz extension
    npz_files = [file for file in files if file.endswith('.pth')]
    
    # Count the number of .npz files
    num_npz_files = len(npz_files)
    
    return num_npz_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=None, help='ID of the scene. Leave empty to infer')
    parser.add_argument('--save_dir', type=str, default='dataset/scenes', help='Directory to save the scenes')
    
    args = parser.parse_args()

    if args.id is None:
        args.id = count_npz_files(args.save_dir) + 1
    
    initArm(True)

    import shutil

    dest = args.save_dir + '/raw_' + str(args.id) + '.pth'
    shutil.move('capture_pointcloud/front_data', dest)

    print('Saved raw data to', dest)

