import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import cv2
import clip

to_tensor = transforms.ToTensor()

def crop(image):
    # https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
    true_points = np.argwhere(image)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    out = image[top_left[0]:bottom_right[0]+1, 
              top_left[1]:bottom_right[1]+1]
    return out

def crop_slice(image):
    # https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
    true_points = np.argwhere(image)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return np.s_[top_left[0]:bottom_right[0]+1, 
              top_left[1]:bottom_right[1]+1]

def crop_slice_dilate(mask, dilate=30):
    # https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
    true_points = np.argwhere(mask)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)

    top_left[0] = max(top_left[0] - dilate, 0)
    top_left[1] = max(top_left[1] - dilate, 0)

    bottom_right[0] = min(bottom_right[0] + dilate, mask.shape[1])
    bottom_right[1] = min(bottom_right[1] + dilate, mask.shape[0])

    out = np.s_[top_left[0]:bottom_right[0]+1, 
              top_left[1]:bottom_right[1]+1]
    return out

def hsv_to_rgb(h, s, v):
    '''
    https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    '''
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

import torchvision.transforms.functional as F


class Resize_with_pad:
    def __init__(self, w=224, h=224):
        self.w = w
        self.h = h

    def __call__(self, image):

        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])

resize = Resize_with_pad()

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != 'cuda':
    import sys
    print("Warning: Not using CUDA", file=sys.stderr)



# Load the seg
# print("Loading Segmentor...", end='', flush=True)
# from fastsam import FastSAM, FastSAMPrompt
# fastsam = FastSAM('./weights/FastSAM-x.pt')

def segment(image: Image):
    everything_results = fastsam(image, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(image, everything_results, device=device)
    return prompt_process, prompt_process.everything_prompt()

# # from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").cuda()
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=16,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=0, # 1
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )

from matplotlib import pyplot as plt

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """
    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb

# Util function to display a point cloud
def view_pc(pc, col=None, show=False, title=None):
    # Setting up the figure
    fig = plt.figure()

    if title is not None:
        plt.title(title)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    if col is None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=col)

    # Set the axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

    if show: plt.show()

    

def choose_object(image_color, image_depth, pc, pc_rgb, heatmap, prompt = 'the cup', topk = 3, viz=False):
    # Must be depth image with HWC
    print(image_depth.shape)
    assert(len(image_depth.shape) == 2)
    assert(image_depth.shape[0:2] == heatmap.shape[0:2])
    assert(image_color.shape[-1] == 3)
    assert(image_color.shape[0:2] == heatmap.shape[0:2])

    # cv2.imshow('input', image_color)
    # cv2.imshow('depth', image_depth)
    

    # Simple RGB for displaying
    image = image_color

    raw_display = image.copy()
    
    transparency = 0.2
    raw_display[:, :, 0] = raw_display[:, :, 0] * transparency + \
                           255 * (1 - transparency) * heatmap / np.max(heatmap)
    
    plt.figure()
    plt.imshow(cv2.cvtColor(raw_display, cv2.COLOR_RGB2BGR))
    plt.title("Raw Image")
    plt.figure()
    plt.imshow(heatmap)
    plt.title("Heatmap")
    
    cv2.waitKey(1)
    
    print("Generating Masks...")
    masks_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prompt_process, results = segment(masks_input) # mask_generator.generate(masks_input)
    print("Num masks:", len(results))

    global fastsam
    del fastsam

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Load the model
    print('Loading CLIP...', end='', flush=True)

    # # assert device == 'cuda'
    # print(device, clip.available_models())
    model, preprocess = clip.load('ViT-B/32', device)
    print('Done')

    # Show results
    # cv2.imshow("result", prompt_process.plot_to_result(annotations=results))
    # cv2.waitKey(1)

    results = (results > 0.5).detach().cpu().numpy()
    # results = list(filter(lambda result : result['area'] > 30, results))
    # print(f"Num masks larger than {30}:", len(results))

    # Apply masks to the RGB images
    input_images = []
    rgb_masks = []
    bool_masks = []
    for i, mask in enumerate(results):
        
        assert(mask.dtype == bool)

        bool_masks.append(mask)

        slice = crop_slice(mask)       
        rgb_mask = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)[slice]
        rgb_masks.append(rgb_mask)

        # print(rgb_mask.shape, crop(mask).shape, crop_slice_dilate(mask, dilate=10))
        
        input_images.append(preprocess(resize(Image.fromarray(rgb_mask, mode='RGB'))))

   
    input_images = torch.stack(input_images).to(device=device)
    text = clip.tokenize([prompt]).to(device=device)
    
    # Run model with text prompt, get probabilities for all masks
    logits_per_image, logits_per_text = model(input_images, text)
    probs = logits_per_image.softmax(0).cpu().detach().squeeze().numpy()
    
    for index in range(len(probs)):
        # multiply probs by heatmap_intersection / area
        # Add one for base probability
        overlap_score = 2 + (results[index] * heatmap).sum() / results[index].sum()
        # print('Overlap score:', overlap_score)

        probs[index] *= overlap_score
    
    # Sort by likelihood
    top_k_indices = np.argsort(probs)[-topk:]

    if viz:
        for i, index in enumerate(reversed(top_k_indices)):
            plt.figure()
            plt.imshow(rgb_masks[index])
            plt.title(f'rank{i + 1}; prob {probs[index]:.3f} for \"{prompt}\"')


    best_mask = bool_masks[top_k_indices[-1]]
    slice = crop_slice(best_mask)

    # plt.figure()
    # plt.imshow(best_mask)
    # plt.title("best mask")
    
    # plt.figure()
    # plt.imshow(image_color[slice])
    # plt.title("color slice")

    # plt.figure()
    # plt.imshow((image_depth * best_mask)[slice])
    # plt.title("depth slice")

    # plt.figure()
    # plt.imshow(image_depth)
    # plt.title("deptH")

    # Old slicing: Use this for pixel-perfect ppointcloud
    image_depth_interest = image_depth * best_mask

    # New slicing: Use this for rectangle region around
    # image_depth_interest = np.zeros_like(image_depth)
    # image_depth_interest[slice] = image_depth[slice]

    plt.figure()
    plt.imshow(image_depth_interest[slice])
    plt.title("Depth cropped around object " + str(slice))

    plt.pause(0.001)

    K = np.array([[911.445649104, 0, 641.169], [0, 891.51236121, 352.77], [0, 0, 1]])
    pc, rgb = depth2pc(image_depth_interest, K, rgb=cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)/256)

    tosave = {'RGB': image_color,
              'depth': image_depth,
              'mask': best_mask,
              'pc': pc,
              'pcrgb': rgb}
    
    torch.save(tosave, '/home/corallab/gaze/shared.pth')

    print("Saved the RGBD and mask to /home/corallab/gaze/shared.pth")

    view_pc(pc, col=rgb)

    return tosave


if __name__ == "__main__":

    import gaze_utils.realsense as rs

    tabletop_reader = rs.RSCapture(serial_number='123122060050')

    import time
    time.sleep(1)

    tabletop_color, tabletop_depth, _, _ = tabletop_reader.get_frames(rotate=False, viz=False)

    # tabletop_depth = cv2.imread('test_images/test_depth.jpeg')
    # tabletop_color = cv2.imread('test_images/test_color.jpg')

    assert(tabletop_color.shape[0] == tabletop_depth.shape[0])    
    assert(tabletop_color.shape[1] == tabletop_depth.shape[1])    
    assert(tabletop_color.shape[2] == 3)       
    assert(tabletop_color.dtype == np.uint8)
    assert(len(tabletop_depth.shape) == 2)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tabletop_depth, alpha=0.03), cv2.COLORMAP_JET).astype(np.uint8)
    plt.imshow(tabletop_depth)
    plt.figure()
    plt.imshow(tabletop_color)
    plt.pause(0.5)

    # image_color, image_depth, heatmap, prompt = torch.load('last_gaze_result.pth')

    # image_color = np.asarray(Image.open('test_images/test_color.jpg'))
    # image_depth = np.asarray(Image.open('test_images/test_depth.jpeg').getchannel(0))

    input_dict = choose_object(tabletop_color, tabletop_depth, np.ones_like(tabletop_depth), 'red apple', topk=10)

    from real_exp_guna import main as start_grasp

    start_grasp(input_dict)