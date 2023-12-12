
import gaze_utils.realsense as rs
import cv2

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

import imageio

from PIL import Image
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

import random

# Utilities
from gaze_utils.util import *

from gaze_results_to_heatmap import show_heatmap

random_colors = np.random.randint(0, 255, (1000, 3))

# Set up OpenGL
print("Compiling OpenGL Shader...", end='', flush=True)
from gaze_utils.arrows import render_frame, WIDTH, HEIGHT
print("Done")

print("Constructing Models...", end='', flush=True)
from gaze_utils.headpose import infer, img2pose_model
from gaze_utils.model import model, gaze_device
print("Done")

assert (rs.CAMERA_H == HEIGHT)
assert (rs.CAMERA_W == WIDTH)

from gaze_cam_calibration import transform
from gaze_results_to_heatmap import get_heatmap

# Load Calibration
import yaml
calib_file = 'gaze_cam_calib.yaml'
try:
    with open(calib_file, "r") as file:
        try:
            w, b = yaml.load(file, Loader=yaml.FullLoader)
            print(f'Loaded Camera Calibration Params from {calib_file} \n', w, '\n', b)
            w = torch.Tensor(w)
            b = torch.Tensor(b)
        except yaml.YAMLError as exc:
            print(f'Failed to parse {calib_file}, using default params', exc)
            w = torch.tensor([[1, 0, 0], 
                            [0, 1, 0],
                            [0, 0, 1]], 
                            dtype=torch.float32)
            b = torch.tensor([[0], [0], [0]], 
                            dtype=torch.float32)
except FileNotFoundError:
    print(f"No such file or directory: {calib_file}. Did you run calibration.py yet?")


def main(input_file_path, output_video_path, output_log_path, prompt, enable_tabletop_capture):

    global model
    global img2pose_model
        
    if enable_tabletop_capture:
        tabletop_reader = rs.RSCapture(serial_number='115422250069')

        import time
        time.sleep(4)

        tabletop_color, tabletop_depth, _, _ = tabletop_reader.get_frames(rotate=False, viz=False)

        # tabletop_depth = cv2.imread('test_images/test_depth.jpeg')
        # tabletop_color = cv2.imread('test_images/test_color.jpg')

        assert(tabletop_color.shape[0] == tabletop_depth.shape[0])    
        assert(tabletop_color.shape[1] == tabletop_depth.shape[1])    
        assert(tabletop_color.shape[2] == 3)       
        assert(tabletop_color.dtype == np.uint8)
        assert(len(tabletop_depth.shape) == 2)

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tabletop_depth, alpha=0.03), cv2.COLORMAP_JET).astype(np.uint8)
        # plt.imshow(tabletop_depth)
        # plt.show()

        torch.save((tabletop_color, tabletop_depth), 'tabletop_image.pth')
        cv2.imshow("target image", tabletop_color)

    else:
        tabletop_color = None
        tabletop_depth = None

    if input_file_path == 'cam':
        v_reader = rs.RSCapture(serial_number='819612072388')
        fps = rs.CAMERA_FPS
        num_frames = 99999
    elif input_file_path == '' or input_file_path == 'webcam':
        v_reader = rs.WebCamCapture()
        fps = rs.CAMERA_FPS
        num_frames = 99999
    else:
        ext = os.path.splitext(input_file_path)[1]
        if ext == '.bag':
            v_reader = rs.RSCapture(input_file_path)
            fps = rs.CAMERA_FPS
            num_frames = 99999
        else:
            v_reader = imageio.get_reader(input_file_path)
            fps = v_reader.get_meta_data()['fps']
            num_frames = v_reader.count_frames()

        # Show target image
        # cv2.namedWindow("target image", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("target image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # 

    frame_step = max(int(fps // 8), 1)

    # Video Writer
    out = imageio.get_writer(output_video_path, fps=fps)

    # Output log data
    output_log = {}

    # Used for tracking faces accross frames
    past_frames = {}
    id_num = 0
    tracking_id = dict()
    identity_last = dict()

    stop = 12000
    try:
        for i, image_src in enumerate(tqdm(v_reader, total=min(stop, num_frames))):

            if i < 0:
                continue

            if i >= stop:
                break

            if image_src is None:
                print("Realsense frame grab error, skipping frame")
                continue

            if type(v_reader) == rs.RSCapture:
                color_image, depth_image, depth_frame, color_frame = image_src
            else:
                color_image = image_src
                
            frame = color_image.copy()
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            past_frames[i] = frame

            # plt.figure(0)
            # plt.imshow(prediction)
            # plt.figure(1)
            # plt.hist(prediction)
            # plt.pause(0.0001)
            
            # Find face pose and bbox
            poses, bboxes = infer(frame)
            bbox = []
                
            for bb in bboxes:
                if bb[0] < 0:
                    bb[0] = 0
                if bb[1] < 0:
                    bb[1] = 0

                dilation_factor = (0.45, 0.5, 0.4) # width, forehead, chin
                bb_box = list(dilate_bbox(bb[0:2], bb[2:4], factor=dilation_factor))
                # Limits
                bb_box[0] = int(max(0, bb_box[0]))
                bb_box[1] = int(max(0, bb_box[1]))
                bb_box[2] = int(max(0, bb_box[2]))
                bb_box[3] = int(max(0, bb_box[3]))

                bb_box[0] = int(min(WIDTH, bb_box[0]))
                bb_box[1] = int(min(HEIGHT, bb_box[1]))
                bb_box[2] = int(min(WIDTH, bb_box[2]))
                bb_box[3] = int(min(HEIGHT, bb_box[3]))
                bbox.append(bb_box)

                # mp_drawing.draw_detection(annotated_image, detection)

            # Match bboxes temporally
            identity_next = dict()
            for j in range(len(bbox)):
                bbox_head = bbox[j]

                if bbox_head is None:
                    continue

                id_val = find_id(bbox_head, identity_last)
                if id_val is None: 
                    id_num += 1
                    id_val = id_num
                
                eyes = [(bbox_head[0] + bbox_head[2]) / 2.0, (0.6 * bbox_head[1] + 0.4 * bbox_head[3])]

                pose_rx, pose_ry, pose_rz, pose_tx, pose_ty, pose_tz = poses[j]
                pose_tx, pose_ty, pose_tz = transform(w, b, [pose_tx, pose_ty, pose_tz]).tolist()

                identity_next[id_val] = (bbox_head, eyes, (pose_tx, pose_ty, pose_tz))

            identity_last = identity_next
            tracking_id[i] = identity_last

            # No human
            if i not in tracking_id.keys():
                print("No Face")
                frame = frame.astype(np.uint8)
                out.append_data(frame)
                output_log[i]['frame'] = cv2.resize(frame, dsize=(frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
                return

            frame = frame.astype(float)

            output_log[i] = {}

            # For each human face id_t
            for face_id in tracking_id[i].keys():
                input_image = torch.zeros(7,3,224,224)
                count = 0

                # Look behind 3 seconds
                for j in range(i - 3 * frame_step, i + 1 * frame_step, frame_step):
                    # If the same guy id_t is in the search frames
                    if j in tracking_id and face_id in tracking_id[j]:
                        # Add this frame to data
                        new_im = Image.fromarray(past_frames[j], 'RGB')
                        bbox, eyes, head_pose = tracking_id[j][face_id]
                    else:
                        # Add the current frame
                        new_im = Image.fromarray(frame, 'RGB')
                        bbox, eyes, head_pose = tracking_id[i][face_id]
                        
                    new_im = new_im.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
                    model_in = transforms.ToTensor()(transforms.Resize((224,224))(new_im))
                    input_image[count,:,:,:] = image_normalize(model_in)

                    count += 1
                
                bbox, eyes, head_pose = tracking_id[i][face_id] 
                bbox = np.asarray(bbox).astype(int)

                # imageio.imwrite('temp.jpg', input_image[0].permute(1,2,0).cpu().numpy())

                #

                # plt.imshow(np.moveaxis(dddd, 0, 2))
                # plt.pause(0.0001)


                output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224).to(gaze_device))
                output_gaze = output_gaze.cpu().detach().numpy()
                gaze = spherical2cartesial(output_gaze)
                gaze = gaze.reshape((-1))

                output_log[i][face_id] = (head_pose, gaze)

                # Render Arrow
                eyes = np.asarray(eyes).astype(float)
                eyes[0], eyes[1] = (eyes[0] / float(frame.shape[1]), eyes[1] / float(frame.shape[0]))

                
                img_arrow = render_frame(eyes[0] - 0.5, -eyes[1] + 0.5, -gaze[0], gaze[1], -gaze[2], 0.05)
                binary_img = ((img_arrow[:,:,0] + img_arrow[:,:,1] + img_arrow[:,:,2]) == 0.0).astype(float)
                binary_img = np.reshape(binary_img, (HEIGHT, WIDTH, 1))
                binary_img = np.concatenate((binary_img, binary_img, binary_img), axis=2)
                frame = binary_img * frame + img_arrow * (1 - binary_img)
            
                # bbox[0], bbox[2] = WIDTH * bbox[0] / frame.shape[1],WIDTH*bbox[2]/frame.shape[1]
                # bbox[1], bbox[3] = HEIGHT * bbox[1] / frame.shape[0],HEIGHT*bbox[3]/frame.shape[0]

                frame = frame.astype(np.uint8)
                # Draw face bbox
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), random_colors[face_id].tolist(), thickness=3)
                # Draw eye dot
                # cv2.circle(frame, (int(eyes[0] * frame.shape[1]), int(eyes[1] * frame.shape[0])), 10, (255, 0, 0), -1)
                # Draw head pose
                frame = cv2.putText(frame, str(f'{head_pose[0]:.{2}f} {head_pose[1]:.{2}f} {head_pose[2]:.{2}f}'), (bbox[0],bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                frame = frame.astype(float)

            frame = frame.astype(np.uint8)

            cv2.imshow('gaze', frame)
            cv2.waitKey(1)

            # output_log[i]['frame'] = cv2.resize(frame, dsize=(frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
            # out.append_data(frame)

            show_heatmap(output_log[i], viz=False)

            # cv2.imshow('heatmap', get_heatmap(8))

    except KeyboardInterrupt:
        print("Ctrl C hit, stopping early")
        pass
    
    import gc
    model.cpu()
    img2pose_model.fpn_model.cpu()
    del model
    del img2pose_model.fpn_model

    # print(f'Writing data to {output_log_path} and video to {output_video_path}')
    # torch.save(output_log, output_log_path)
    # print('You might see a segfault. That\'s normal, video writer does that')
    # out.close()

    print("Sending heatmap to segmentation model")

    gc.collect()

    # Last 8 frames
    torch.cuda.empty_cache()

    print((tabletop_color, tabletop_depth, get_heatmap(8), prompt))

    torch.save( (tabletop_color, tabletop_depth, get_heatmap(8), prompt), 'last_gaze_result.pth')

    # from gaze_heatmap_to_mask import choose_object
    # choose_object(tabletop_color, tabletop_depth, get_heatmap(8), prompt=prompt, topk=3)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='') # '/mnt/c/Users/Lilith/Documents/calib2.bag'
    parser.add_argument('-o', '--output', type=str, default='gaze_visualize.mp4')
    parser.add_argument('-l', '--log', type=str, default='gaze_tracking_results.pt')
    parser.add_argument('-p', '--prompt', type=str, default='the banana')
    parser.add_argument('-t', '--disable_tabletop_cam', action='store_false')

    args = parser.parse_args()

    main(args.input, args.output, args.log, args.prompt, not args.disable_tabletop_cam)