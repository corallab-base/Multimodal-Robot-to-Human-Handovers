# Imports
import sys
sys.path.append('../')

import time
import cv2
import torch
import matplotlib.pyplot as plt
from gaze_utils.realsense import RSCapture
from gaze_utils.headpose import infer, free_model

# Set the seed for reproducibility
import random
random.seed(42)
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


# Gets the gaze heatmap
def gaze_track(time_to_wait, display_image):
    v_reader = RSCapture(serial_number='123122060050', dont_set_depth_preset=True)

    heatmap = None
    frame_num = 0
    for frame in v_reader:
        color_image, depth_image, depth_frame, color_frame = frame

        # Run headpose tracking
        heatmap = infer(color_image, display_image, viz=True)

        if frame_num == time_to_wait * 15:
            # free_model()
            cv2.destroyAllWindows()
            plt.imshow(heatmap, cmap='binary', interpolation='nearest')
            plt.show()
            return heatmap
        
        frame_num += 1


# Unpacks the stored scene data
def get_scene_data(index):
    data = torch.load(f'./scenes/scene_{index + 1}.pth')
    front_rgb, object_positions = data['front_rgb'], data['object_positions']
    return front_rgb, object_positions

scene_chosen = []

def viz_pos(scene_index, rand_obj_index, positions, selected_obj):
    plt.cla()
    # Visualizing the generated scenes
    for index, (name, (x, y)) in enumerate(positions):
        if index == selected_obj:
            plt.scatter(x, y, s=2000, color='red')
            plt.text(x, y, name, rotation=30, fontsize=10)
            scene_chosen.append(name)
            print('chosen', name)
            
        else:
            plt.scatter(x, y, s=2000, color='green')
            plt.text(x, y, name, rotation=30, fontsize=10)

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
    # plt.show()

    plt.savefig(f'chosen_images/img_{scene_index+1}_{rand_obj_index}.png')


def main():
    # Current participant
    part_num = 2

    # Generates a "random" 2D array
    obj_selection, size = [], 50
    for _ in range(size):
        first_num = random.randint(0, 6)
        second_num = random.randint(0, 6)
        
        # Ensure second number is different from the first
        while second_num == first_num:
            second_num = random.randint(0, 6)
    
        first_num = min(first_num, 5)
        second_num = min(second_num, 5)
        obj_selection.append([first_num, second_num])
    

    # Going through frame by frame
    for scene_index in range(0, 50):
        print(scene_index)

        # Getting the scene info
        front_rgb, object_positions = get_scene_data(scene_index)

        for rand_obj_index in [0, 1]: 
            # Visualize the objects
            # print(obj_selection[scene_index][rand_obj_index])
            viz_pos(scene_index, rand_obj_index, object_positions, obj_selection[scene_index][rand_obj_index])

            # # Getting the heatmap
            # time_to_wait = 2
            # heatmap = gaze_track(time_to_wait, front_rgb)

            # # Saving the heatmap
            # torch.save(heatmap, f'part_gaze/part{part_num}/heatmap_{scene_index + 1}_q{rand_obj_index}.pth')

        with open('scene_chosen.txt', 'w') as file:
            for item in scene_chosen:
                file.write(f"{item}\n")

if __name__ == "__main__":
    main()
    # data = torch.load('part_gaze/part1/heatmap_1_q0.pth')
    # plt.imshow(data, cmap='binary', interpolation='nearest')
    # plt.show()