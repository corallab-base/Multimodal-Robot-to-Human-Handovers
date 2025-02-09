
import argparse
import shutil

parser = argparse.ArgumentParser(description='Copy file to destination directory.')
parser.add_argument('--index', type=int, required=True, help='Index of the file to copy')

args = parser.parse_args()

source_path = f'dataset/scenes/raw_{args.index}.pth'
destination_path = 'capture_pointcloud/front_data'
shutil.copy(source_path, destination_path)

from mover import moveit

objname = 'bottle'

# pc_dest = f'dataset/scenes/raw_pc_{objname}.pth'
moveit(objname=objname, partname=None, target_holder=None, save_pc=True)
        
