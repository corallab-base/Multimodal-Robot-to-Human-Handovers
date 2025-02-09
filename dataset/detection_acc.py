import os
import cv2
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gaze_utils.sshlib import get_glip

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def on_continue(event):
    plt.close() 

def display_images_with_button(image_path1, image_path2):
    # Load images

   
    img1 = cv2.cvtColor(image_path1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image_path2, cv2.COLOR_BGR2RGB)

    # print(img1.dtype, img2.dtype)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

    # Display images
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    
    # Show the plot
    

def convert_darknet_bbox_to_standard(bb, img):
    """
    Convert Darknet bounding box format to standard format.

    Parameters:
        x_center (float): Normalized x center of the bounding box
        y_center (float): Normalized y center of the bounding box
        width (float): Normalized width of the bounding box
        height (float): Normalized height of the bounding box
        img_width (int): Width of the image
        img_height (int): Height of the image

    Returns:
        tuple: (x1, y1, x2, y2) where x1, y1 are the top-left coordinates and
               x2, y2 are the bottom-right coordinates of the bounding box.
    """
    x_center, y_center, width, height = bb
    # Convert from normalized values to image dimensions
    assert img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]
    img_height, img_width, _ = img.shape 
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate coordinates of the top left corner
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)

    # Calculate coordinates of the bottom right corner
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return (x1, y1, x2, y2)


def draw_bounding_boxes(image, bounding_boxes, labels=None):
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw each bounding box
    for i, bbox in enumerate(bounding_boxes):
        # Each bbox is expected to be a tuple (x, y, width, height)
        x, y, x2, y2 = bbox
        # Create a rectangle patch
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=1, edgecolor=['green', 'red'][i], facecolor='none')
        if labels is not None:
            ax.annotate(labels[i], (x, y-16), color='white', weight='bold', fontsize=8,
                        xytext=(0, -10), textcoords='offset pixels', ha='left', backgroundcolor='red')

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Display the plot with the bounding boxes
    plt.title("Selected (Green) vs Gt (red)")

    plt.subplots_adjust(bottom=0.2)

    # Add a continue button
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # x, y, width, height
    button = Button(ax_button, 'Continue')
    button.on_clicked(on_continue)
    
    
    plt.show()

def parse_darknet_labels(file_path):
    labels = []
    print('parse_darknet_labels', file_path)
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            # print(class_id)
            if class_id == 1 or class_id == 2:
                labels.append((class_id, x_center, y_center, width, height))
    return labels


def closest_bb(heatmap, bounding_boxes):
    if len(bounding_boxes) == 0:
        print('No BB available from glip')
        return None
    
    closest_box = None
    min_fill = 1000000000
    for bb in bounding_boxes:
        x1, y1, x2, y2 = bb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        fill = heatmap[y1:y2, x1:x2].sum() / ((y2 - y1) * (x2 - x1))

        if fill < min_fill:
            min_fill = fill
            closest_box = bb
    return closest_box

def calculate_iou(box1, box2):

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def find_file_starting_with(prefix, directory):
    # List all files in the directory
    # print('listdir', directory, prefix)
    for file in os.listdir(directory):
        # print('consider', file, prefix in str(file), 'txt' in str(file))
        # Check if the file name starts with the given prefix
        if prefix in str(file) and 'txt' in str(file):
            return os.path.join(directory, file)
    return None  # Return None if no file matches

def parse_prompts(pompt_path):
    d = torch.load(pompt_path)
    # print('loaded', pompt_path, d['object_positions'], d['queries'])
    return d['front_rgb'], d['object_positions'][0][0], d['object_positions'][1][0]

def process_scene(ii, prompt_path, label_path):
    
    ground_truth_labels = parse_darknet_labels(find_file_starting_with(label_path, 'dataset/labels/train/'))
    
    image, name0, name1 = parse_prompts(prompt_path)

    name0 = lines[ii*2]
    name1 = lines[ii*2+1]
    print('prompt:', name0, ',', name1)

    bounding_boxes1, ann1 = get_glip(name0, image)
    # draw_bounding_boxes(image, bounding_boxes1)
    bounding_boxes2, ann2 = get_glip(name1, image) 
    # draw_bounding_boxes(image, bounding_boxes2)

    print('glip results', bounding_boxes1, bounding_boxes2)

    display_images_with_button(ann1, ann2)
    # Adjust layout
    plt.subplots_adjust(bottom=0.2)

    # Add a continue button
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # x, y, width, height
    button = Button(ax_button, 'Continue')
    button.on_clicked(on_continue)
    plt.title(f'GLIP candidates for each object {name0}, {name1}')
    plt.show()

    results = []

    for q, bounding_boxes in enumerate([bounding_boxes1, bounding_boxes2]):
        ious = []
        for i in range(1, 4):  # For 3 participants
            print(f'    object {q + 1}, participant', i)
            heatmap_path = torch.load(f"dataset/part_gaze/part{i}/heatmap_{ii+1}_q{q}.pth")
            selected_bb = closest_bb(heatmap_path, bounding_boxes)
            # print('closest bb of', bounding_boxes, ':', selected_bb)

            if selected_bb is not None:

                # Assuming the ground truth is always available and matches one entry in labels
                ground_truth = ground_truth_labels[q][1:]  # Simplified selection of ground truth

                labels = [[name0, name1][q], '']
                draw_bounding_boxes(image, [selected_bb, convert_darknet_bbox_to_standard(ground_truth, image)], labels)

                selected_iou = calculate_iou(selected_bb, convert_darknet_bbox_to_standard(ground_truth, image))
                ious.append(selected_iou)

        if len(ious) == 0:
            results.append(0)
        else:
            iou = sum(ious) / len(ious)
            results.append(iou)

with open('dataset/scene_chosen.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]

for i in range(18, 51):
    process_scene(i, f'dataset/scenes/scene_{i+1}.pth', f'scene_{i + 1}_png')
