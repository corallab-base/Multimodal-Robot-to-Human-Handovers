# Imports
import random
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import torchvision.transforms.functional as F
from gaze_utils.sshlib import get_glip


obj_parts = {
    'scissors': ['handle', 'tip', 'blade'],
    'extra large clamp': ['handle'],
    'bowl': ['lip'],
    'mug': ['handle', 'lip'],
    'apple': [],
    'tennis ball': [],
    'pear': [],
    'lemon': [],
    'racquetball': [],
    'baseball': [],
    'cup': ['lip'],
    'strawberry': [],
    'colored wood block': [],
    'phillips screwdriver': ['tip', 'handle'],
    'flat screwdriver': ['tip', 'handle'],
    'orange': [],
    'banana': [],
    'rubik\'s cube': [],
    'bleach cleanser': ['tip'],
    'master chef can': ['lip'],
    'potted meat can': ['tip'],
    'tuna fish can': ['lip'],
    'tomato soup can': ['lip'],
    'chips can': [],
    'mustard bottle': ['tip']
}


# Reading in all the scene information
def get_scene_images():
    total_scenes=50
    for scene_index in range(0,total_scenes):
        print(f'Scene: {scene_index}')
        scene_data = torch.load(f'scenes/scene_{scene_index + 1}.pth')
        front_rgb = scene_data['front_rgb']
        front_rgb[:, :, [0, 2]] = front_rgb[:, :, [2, 0]]

        pil_image = F.to_pil_image(front_rgb)
        pil_image.save(f'dataset/scenes_images/scene_{scene_index + 1}.png')


def read_json():
    with open('dataset/annotations/_annotations.coco.json', 'r') as file:
        # Load the JSON data into a dictionary
        data_dict = json.load(file)

    id_to_scene_ind, image_data = {}, data_dict['images']
    for item in image_data:
        id, scene_index = item['id'], int(item['file_name'][6:item['file_name'].rfind('_')]) - 1
        id_to_scene_ind[id] = scene_index

    bb_lists, annots = defaultdict(list), data_dict['annotations']
    for item in annots:
        id = int(item['image_id'])
        bb_lists[id_to_scene_ind[id]].append((data_dict['categories'][item['category_id']]['name'], item['bbox']))

    return bb_lists


# Reads the gaze file
def get_gaze(part, scene_index, rand_obj_ind):
    heatmap = torch.load(f'dataset/part_gaze/part{part}/heatmap_{scene_index + 1}_q{rand_obj_ind}.pth')
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    heatmap = cv2.resize(heatmap, None, fx=8, fy=8,
                                   interpolation=cv2.INTER_NEAREST)
    x, y = 8 * x, 8 * y

    # Creating a bitmask and upsampling
    threshold = .2
    bitmask = (heatmap > threshold).astype(np.uint8)

    return x, y, bitmask, heatmap


def get_distance(r_x, r_y, r_w, r_h, x, y):
    # Gets distance to the closest part of the bounding box
    d_top, d_bottom = abs(y - r_y), abs(y - (r_y + r_h))
    corner_y = r_y if d_top < d_bottom else (r_y + r_h)
    
    d_left, d_right = abs(x - r_x), abs(x - (r_x + r_w))
    corner_x = r_x if d_left < d_right else (r_x + r_w)
    
    d_cx, d_cy = (x - corner_x), (y - corner_y)
    d_corner = np.sqrt(d_cx**2 + d_cy**2)

    return min(d_corner, d_top, d_bottom, d_left, d_right)

    # Gets distance to the center of the object
    rc_x, rc_y = (r_x + (r_w / 2)), (r_y + (r_h / 2))
    d_cx, d_cy = (x - rc_x), (y - rc_y)
    return np.sqrt(d_cx ** 2 + d_cy ** 2)


def in_box(r_x, r_y, r_w, r_h, x, y):
    return (x >= r_x) and (x <= (r_x + r_w)) and (y >= r_y) and (y <= (r_y + r_h))


# Checks whether the correct object is chosen and MSE
def metrics(bb_list, gaze_x, gaze_y, bitmask, correct_category):
    best_idx, min_dist, correct_idx = -1, float('inf'), -1
    for index, box in enumerate(bb_list):
        x, y, width, height = box[1]
        if in_box(x, y, width, height, gaze_x, gaze_y):
            dist = 0
        else:
            dist = get_distance(x, y, width, height, gaze_x, gaze_y)

        if dist < min_dist:
            min_dist = dist
            best_idx = index

        if box[0] == correct_category:
            correct_idx = index

    correct = (best_idx == correct_idx)

    # Calculating the IoU scores
    bbox_bitmask = np.zeros_like(bitmask, dtype=np.uint8)
    x, y, width, height = bb_list[correct_idx][1]
    bbox_bitmask[y:y+int(height), x:x+int(width)] = 1
    intersection = np.logical_and(bitmask, bbox_bitmask).astype(np.uint8).sum()
    union = np.logical_or(bitmask, bbox_bitmask).astype(np.uint8).sum()
    iou = intersection / union if union > 0 else 0.0

    return correct, min_dist, iou


def get_gaze_metrics(bb_lists, viz=False):
    # Generating the metrics
    for part in [1, 2, 3]:
        metrics_acc = []
        for scene_index in range (0, 50):
            for rand_obj_ind in [0, 1]:
                # Getting the necessary information
                x, y, bitmask, heatmap = get_gaze(part, scene_index, rand_obj_ind)

                if viz:
                    scene_data = torch.load(f'dataset/scenes/scene_{scene_index + 1}.pth')
                    front_rgb = scene_data['front_rgb']
                    front_rgb[:, :, [0, 2]] = front_rgb[:, :, [2, 0]]

                    plt.imshow(front_rgb)
                    plt.axis('off')
                    plt.scatter(x, y, color='red', marker='o')

                    for bbox in bb_lists[scene_index]:
                        x_min, y_min, width, height = bbox[1]

                        rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        plt.gca().add_patch(rect)

                    plt.show()

                rel_bb_list = bb_lists[scene_index]

                # Getting the metrics
                correct_category = 'one' if rand_obj_ind == 0 else 'two'
                correct, mse, iou = metrics(rel_bb_list, x, y, bitmask, correct_category)
                metrics_acc.append([correct, mse, iou])

                if viz:
                    print(f'Part {part}, Scene {scene_index}: {correct}, {mse}, {iou}')


        # Metrics summary
        correctness, mses, ious = [], [], []
        for index in range (len(metrics_acc)):
            correctness.append(metrics_acc[index][0])
            if correctness[-1]:
                mses.append(metrics_acc[index][1])
                ious.append(metrics_acc[index][2])

        for obj_ind in range (0, 2):
            print(f'For participant #{part} and object #{obj_ind}:')

            correct_sub, mses_sub, ious_sub = correctness[obj_ind::2], mses[obj_ind::2], ious[obj_ind::2]
            print(f'Number of successful object selections: {sum(correct_sub) / len(correct_sub)}')
            print(f'Average distance to closest object: {np.average(mses_sub)}')
            print(f'Average IoU scores: {np.average(ious_sub)}\n')


def run_glip():
    # Loading in the objects of interest for each scene
    obj_names = []
    with open("dataset/scene_chosen.txt") as f:
        for line in f:
            obj_names.append(line)


    # Loading in the glip heatmaps
    glip_heatmaps = {}
    for scene_index in range (0, 50):
        image = f'dataset/scenes_images/scene_{scene_index + 1}.png'
        image = plt.imread(image)
        image = (image * 255).astype(np.uint8)
        for query_index, object_name in enumerate(obj_names[2 * scene_index: (2 * scene_index) + 2]):
            object_name = object_name.replace("\n", "").replace("\'", "")
            print(object_name)
            bbox, ann_image, score = get_glip(object_name, image)

            # plt.title(object_name)
            # plt.imshow(ann_image)
            # plt.show()

            glip_heatmaps[(scene_index, query_index)] = bbox, score, ann_image
    
    # Saving the glip heatmaps
    # torch.save(glip_heatmaps, f'dataset/glip_heatmaps.pth')


def get_glip_metrics(bb_lists):
    # Restoring the glip heatmaps
    glip_heatmaps = torch.load(f'dataset/glip_heatmaps.pth')

    # Getting GLIP standalone results
    correct, total_dist, total_trails = [], [], 0
    correct, total_dist = [], []
    for scene_index in range (0, 50):
        for query_index in range(2):
            # Finding the correct category
            correct_category = 'one' if query_index == 0 else 'two'

            # Retreving boxes and their scores
            gh, score, ann_image = glip_heatmaps[(scene_index, query_index)]

            # Case where gaze failed
            if len(gh) == 0:
                correct.append(False)
                total_dist.append(0)
                continue

            # print(score)
            # plt.imshow(ann_image)
            # plt.show()

            total_trails += 1

            # Getting best bouding box
            best_bbox = gh[np.argmax(score)]
            glip_x, glip_y = (best_bbox[0] + best_bbox[2]) / 2, (best_bbox[1] + best_bbox[3]) / 2

            best_idx, min_dist, correct_idx = -1, float('inf'), -1
            for index, box in enumerate(bb_lists[scene_index]):
                x, y, width, height = box[1]
                if in_box(x, y, width, height, glip_x, glip_y):
                    dist = 0
                else:
                    dist = get_distance(x, y, width, height, glip_x, glip_y)

                if dist < min_dist:
                    min_dist = dist
                    best_idx = index

                if box[0] == correct_category:
                    correct_idx = index

            correct.append(best_idx == correct_idx)
            total_dist.append(dist)

    # Printing out the results
    print(f'Accuracy: {np.sum(correct) / len(correct)}')
    print(f'MSE: {np.sum(total_dist) / total_trails}')


def get_gaze_glip_results(bb_lists):
    # Restoring the glip heatmaps
    glip_heatmaps = torch.load(f'dataset/glip_heatmaps.pth')

    for part in [1, 2, 3]:
        metrics_acc = []
        for scene_index in range(50):
            for query_ind in range(2):
                # Getting the heatmaps
                x, y, bitmask, heatmap = get_gaze(part, scene_index, query_ind)
                bbox, score, ann_image = glip_heatmaps[(scene_index, query_ind)]

                # Bad glip heatmap
                glip = True
                if len(bbox) == 0:
                    glip = False
                    continue

                # Combining the heatmaps
                if glip:
                    intersection_scores = []
                    for bb in bbox:
                        x1, y1, x2, y2 = bb
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                        area = ((y2 - y1) * (x2 - x1))
                        intersection_scores.append(heatmap[y1:y2, x1:x2].sum() * 1/area)

                    # Get best object regions
                    best_obj = bbox[np.argmax(intersection_scores)]

                    # Get object region
                    x1, y1, x2, y2 = best_obj
                    x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Getting the metrics
                correct_category = 'one' if query_ind == 0 else 'two'
                correct, mse, iou = metrics(bb_lists[scene_index], x, y, bitmask, correct_category)
                metrics_acc.append([correct, mse, iou])

        # Metrics summary
        correctness, mses, ious = [], [], []
        for index in range (len(metrics_acc)):
            correctness.append(metrics_acc[index][0])
            if correctness[-1]:
                mses.append(metrics_acc[index][1])
                ious.append(metrics_acc[index][2])

        print(f'For participant #{part}:')
        print(f'Number of successful object selections: {sum(correctness) / len(correctness)}')
        print(f'Average distance to closest object: {np.average(mses)}')
        print(f'Average IoU scores: {np.average(ious)}\n')


def run_glipparts():
    # Loading in the objects of interest for each scene
    obj_names = []
    with open("dataset/scene_chosen.txt") as f:
        for line in f:
            obj_names.append(line)

    # Restoring the glip heatmaps
    glip_heatmaps = torch.load(f'dataset/glip_heatmaps.pth')

    # Getting GLIP standalone results
    trials, npp = 0, 0
    for scene_index in range (0, 50):
        for query_index in range(2):
            # Finding the correct category
            correct_category = 'one' if query_index == 0 else 'two'

            # Retreving boxes and their scores
            gh, score, ann_image = glip_heatmaps[(scene_index, query_index)]

            # Case where gaze failed
            if len(gh) == 0:
                continue

            # Getting best bouding box
            best_bbox = gh[np.argmax(score)]
            glip_x, glip_y = (best_bbox[0] + best_bbox[2]) / 2, (best_bbox[1] + best_bbox[3]) / 2

            best_idx, min_dist, correct_idx = -1, float('inf'), -1
            for index, box in enumerate(bb_lists[scene_index]):
                x, y, width, height = box[1]
                if in_box(x, y, width, height, glip_x, glip_y):
                    dist = 0
                else:
                    dist = get_distance(x, y, width, height, glip_x, glip_y)

                if dist < min_dist:
                    min_dist = dist
                    best_idx = index

                if box[0] == correct_category:
                    correct_idx = index

            correct = (best_idx == correct_idx)
            
            if correct:
                trials += 1
                # Checking if there are grasps
                obj = obj_names[2*scene_index + query_index]
                parts = obj_parts[obj.strip()]
                if parts == []:
                    npp += 1
                    continue

                random_number = 2
                if random_number == 1:
                    # No part specified
                    pass
                elif random_number == 2:
                    # Robot part specified
                    # Getting the image
                    image = f'dataset/scenes_images/scene_{scene_index + 1}.png'
                    image = plt.imread(image)
                    image = (image * 255).astype(np.uint8)

                    # Random part specified
                    rand_part = random.choice(parts)

                    # Getting GLIP part
                    bbs_part, ann_part, scores = get_glip(rand_part, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if scores.size == 0:
                        print("Fail!")
                        continue

                    # Getting best bouding box
                    best_bbox = bbs_part[np.argmax(scores)]
                    print(obj, parts, rand_part)
                    x_min = int(best_bbox[0])
                    y_min = int(best_bbox[1])
                    x_max = int(best_bbox[2])
                    y_max = int(best_bbox[3])
                    width = x_max - x_min
                    height = y_max - y_min

                    # Visualizing the chosen part
                    plt.imshow(image)
                    plt.axis('off')
                    
                    rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.show()
                    print("Success!")
                elif random_number == 3:
                    # Human part specified
                    pass
                else:
                    # Both human and robot part specified
                    pass
    print(trials, npp)


if __name__ == "__main__":
    viz = False
    bb_lists = read_json()

    # Getting gaze results
    # print("Gaze metrics:")
    # get_gaze_metrics(bb_lists, viz)
    print("\n\n")

    # Running the glip models
    # run_glip()

    # Getting object selection metrics
    # print("GLIP Metrics")
    # get_glip_metrics(bb_lists)
    # print("\n\n")

    # # Getting the combined results
    # print("Gaze + GLIP Metrics")
    # get_gaze_glip_results(bb_lists)
    # print("\n\n")

    run_glipparts()