import multiprocessing
import os
import time

import numpy as np
import torch

def get_best_grasp(pred_grasps_cam, scores, hand_pcs):
    '''
    Select pred_grasps_cam with best score

    pred_grasps_cam (NxMxGrasp) is a list of arrays containing grasps
    scores (NxMxScores) is a list of arrays containing scores

    Note M can be empty

    if hand_pcs, hand_cols is not empty, select farthest
    '''
    avoid_hands = hand_pcs is not None

    # to list of ndarrays
    pred_grasps_cam = pred_grasps_cam.item()
    scores = scores.item()

    # Flatten to {score: pose}
    store = {}
    store2 = {}
    points = []
    for key in pred_grasps_cam:
        for score, gpose in zip(scores[key], pred_grasps_cam[key]):
            store[score] = gpose
            points.append(gpose[0:3, 2])
            store2[tuple(gpose[0:3, 2])] = gpose
    points = np.array(points)

    print('avoid hands disabled')
    if avoid_hands and False:
        hand_pc = np.concatenate(hand_pcs, axis=0)

        assert points.shape[1] == 3, 'points not Nx3' + str(points)
        assert hand_pc.shape[1] == 3, 'hand_pc not Nx3' + str(hand_pc)

        print('Hand and PC shapes:', hand_pc.shape, points.shape)

        dists = torch.cdist(torch.from_numpy(points).float(), torch.from_numpy(hand_pc).float())

        max_index = (dists == torch.max(dists)).nonzero()

        # print('max_index', max_index)

        # max_index has a size 1 dim1
        return store2[
            tuple(points[
                    max_index[0][0]
                ])
            ]

    # Get highest score in pred_grasps_cam
    if len(store) == 0:
        raise Exception("No contact-graspnet candidates")
    best_score = max(store.keys())
    best_grasp = store[best_score]

    # print("Full best grasp (cam space)", 'score:', best_score, ''.join(['\n   ' + s for s in str(best_grasp).split('\n')]))

    # Extract translation and rotations
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(best_grasp[0:3, 0:3])
    trans = np.array([best_grasp[0:3, 3]])

    return trans, rot


def best_grasp(pc_full, pred_grasps_cam, scores, contact_pts, pc_colors, hand_pcs, hand_cols, avoid_hands):

    # from grasp_vis import visualize_grasps
    # from threading import Thread

    # # Run visualization in a another thread since it takes a while
    # print('Visualizing Grasps...takes time')
    # multiprocessing.Process(target=visualize_grasps,
    #         args=(pc_full, pred_grasps_cam.item(), scores.item(), hand_pcs, hand_cols), 
    #        kwargs={'plot_opencv_cam': False, 'pc_colors': pc_colors}).start()
    
    try: os.remove('grasp_vis_data.pickle')
    except FileNotFoundError: pass

    import pickle

    np.savez('grasp_vis_data.npz', pc_full, pred_grasps_cam, scores, hand_pcs, hand_cols, pc_colors)
    
    return get_best_grasp(pred_grasps_cam, scores, hand_pcs if avoid_hands else None)

if __name__ == '__main__':
    from grasp_vis import visualize_grasps
    from threading import Thread
    import pickle

    # Run visualization in a another thread since it takes a while
    print('Vis grasps')
    data = np.load('grasp_vis_data.npz', allow_pickle=True)
    pc_full, pred_grasps_cam, scores, hand_pcs, hand_cols, pc_colors = [data[name] for name in data.files]

    visualize_grasps(pc_full, pred_grasps_cam.item(), scores.item(), hand_pcs, hand_cols, 
           plot_opencv_cam=False, pc_colors=pc_colors)