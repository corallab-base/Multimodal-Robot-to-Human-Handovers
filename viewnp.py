import numpy as np

d = np.load('tmp/grasp_res.npz', allow_pickle=True)

print(list(d.keys()))

print(d['pred_grasps_cam'].item())