
# Launch the thing
import os
import subprocess
import time
import numpy as np
import paramiko
from gaze_utils.constants import d415_intrinsics

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

k = paramiko.Ed25519Key.from_private_key_file('secrets/id_ed25519')

sftp = None

def connect():
    global sftp

    print('SSH connect to corallab server...')
    ssh.connect(hostname='10.164.8.169', username='corallab-s1', pkey=k)
    sftp = ssh.open_sftp()
    print('    ...Connected!')


def get_grasp(rgb, depth, mask, avoid_hands):
    '''
    Find a grasp in mask, avoid hand candidates if true
    '''
    print("Requesting HUMAN and ROBOT Grasp position")

    unique = str(int(100 * time.time()))

    np.savez('tmp/grasp' + unique + '.npz', depth=depth, rgb=rgb, segmap=mask, K=d415_intrinsics)
    
    # for robot grasp gen
    send('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/INPUT' + unique + '.npz')
    if avoid_hands:
        # for human grasp gen
        send('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/INPUT' + unique + '.npz')
    
    # Send DONE
    send('tmp/DONE', 
         '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/DONE')
    
    if avoid_hands:
        send('tmp/DONE', 
             '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/DONE')

    # Wait for robot grasp to come through
    for _ in range(100):
        succ = get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/predictions_INPUT' + unique + '.npz', 
                   'tmp/grasp_res.npz')
        
        if succ: break
        print('    wait for ROBOT GRASP')
        time.sleep(0.2)
    else:
        raise Exception('Did not get ROBOT GRASP result in 20s')
    
    print("    Got robot grasp position")
    
    data = np.load('tmp/grasp_res.npz', allow_pickle=True)
    pc_full, pred_grasps_cam, scores, contact_pts, pc_colors = \
        data['pc_full'], data['pred_grasps_cam'], data['scores'], data['contact_pts'], data['pc_colors']
    
    if avoid_hands:
        # Wait for human grasp to come through
        for _ in range(100):
            succ = get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/hand_predictions_INPUT' + unique + '.npz', 
                       'tmp/grasp_hand_res.npz')
            
            if succ: break
            print('    wait for HAND POSITION')
            time.sleep(0.2)
        else:
            raise Exception('Did not get the HUMAN HAND result in 20s')
    
        print("    Got human hand position")

        dataa = np.load('tmp/grasp_hand_res.npz', allow_pickle=True)
        hand_pcs, hand_cols = \
            dataa['hand_pcs'], dataa['hand_cols']
    else:
        hand_pcs, hand_cols = None, None

    return pc_full, pred_grasps_cam, scores, contact_pts, pc_colors, hand_pcs, hand_cols

def send(src, dest):
    try:
        command = 'scp -i secrets/id_ed25519 ' + src + ' corallab-s1@10.164.8.169:' + dest
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=2)

        if not result.returncode == 0:
            print('send error ', command, '\n    ', result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    
def get(src, dest):
    try:
        command = 'scp -i secrets/id_ed25519 corallab-s1@10.164.8.169:' + src + ' ' + dest
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=2)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def get_glip(prompt, image):

    print("Requesting GLIP result")

    from PIL import Image
    import time

    imgname = 'glipimage' + str(int(100 * time.time())) + '-' + str(prompt.replace(' ', '_'))
    Image.fromarray(image).save('tmp/' + imgname + '.png')
   
    send('tmp/' + imgname + '.png', 
         '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/' + imgname + '.png')
    send('tmp/DONE', 
         '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/DONE')

    for _ in range(20):
        succ = get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/outs/' + imgname + '.npz',
                   ' tmp/dest.npz')

        if succ: break

        time.sleep(0.2)
    else:
        raise Exception("GLIP timed out 10s")

    os.remove('tmp/' + imgname + '.png')

    # sftp.remove(f'/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/outs/' + imgname + '.npz')

    dat = np.load('tmp/dest.npz', allow_pickle=True)
    ann_image, bbox, score, labels = dat['ann_image'], dat['bbox'], dat['score'], dat['labels']

    print('    Got GLIP result')

    # from matplotlib import pyplot as plt
    # plt.imshow()
    # plt.figure()
    # plt.imshow(image)
    # plt.pause(0.1)

    return bbox, ann_image