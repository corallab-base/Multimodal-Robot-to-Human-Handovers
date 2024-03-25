import numpy as np
import torch

def spherical2cartesial(x): 
    output = np.zeros((x.shape[0],3))
    output[:,2] = -np.cos(x[:,1]) * np.cos(x[:,0])
    output[:,0] = np.cos(x[:,1]) * np.sin(x[:,0])
    output[:,1] = np.sin(x[:,1])
    return output

def get_head_bbox(body_bbox, dense_pose_record):
    #Code from Detectron modifyied to make the box 30% bigger
    """Compute the tight bounding box of a binary mask."""
    # Head mask
    labels = dense_pose_record.labels.cpu().numpy()
    mask = (labels == 23) | (labels == 24)    
    if not np.any(mask):
      return None

    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    W = labels.shape[0]
    H = labels.shape[1]
    
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]

    y0 = ys[0]
    y1 = ys[-1]
    w = x1-x0
    h = y1-y0
    
    x0 = max(0,x0-w*0.15)
    x1 = max(0,x1+w*0.15)
    y0 = max(0,y0-h*0.15)
    y1 = max(0,y1+h*0.15)

    x0 += body_bbox[0]
    x1 += body_bbox[0]
    y0 += body_bbox[1]
    y1 += body_bbox[1]

    return np.array((x0, y0, x1, y1), dtype=np.float32)


def extract_heads_bbox(frame):
    N = len(frame['pred_densepose'])
    if N==0:
      return []
    bbox_list = []
    for i in range(N-1,-1,-1):
      dp_record = frame['pred_densepose'][i]
      body_bbox = frame['pred_boxes_XYXY'][i].cpu().numpy()
      bbox = get_head_bbox(body_bbox, dp_record)
      if bbox is None: 
        continue
      bbox_list.append(bbox)
    return bbox_list

def compute_iou(bb1,bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    bb2_area = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    eps = 1e-8

    if iou <= 0.0  or iou > 1.0 + eps: return 0.0

    return iou

# Dilate the bbox
# Tweaked to match detectron's DensePose BBox as much as possible
def dilate_bbox(rect_start, rect_end, factor=0.15):
  siz = (rect_end[0] - rect_start[0], rect_end[1] - rect_start[1])

  if type(factor) == float or type(factor) == int:
     factor = (factor, factor, factor)

  # In order: left, top, down
  return rect_start[0] - siz[0] * factor[0], \
         rect_start[1] - siz[1] * factor[1], \
         rect_end[0] + siz[0] * factor[0], \
         rect_end[1] + siz[1] * factor[2] \
    
def find_id(bbox,id_dict):
    id_final = None
    max_iou = 0.5
    for k in id_dict.keys():
        if(compute_iou(bbox,id_dict[k][0])>max_iou): 
            id_final = k
            max_iou = compute_iou(bbox,id_dict[k][0])
    return id_final