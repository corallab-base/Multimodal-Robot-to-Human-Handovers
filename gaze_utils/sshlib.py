
# Launch the thing
import os
import subprocess
import time
import numpy as np
import paramiko
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

    np.savez('tmp/grasp' + unique + '.npz', depth=depth, rgb=rgb, segmap=mask, K=[911.445649104, 0, 641.169, 0, 891.51236121, 352.77, 0, 0, 1])
    
    # Save this for robot grasp gen
    # sftp.put('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/INPUT' + unique + '.npz')
    send('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/INPUT' + unique + '.npz')
    if avoid_hands:
        # Save this for human grasp gen
        # sftp.put('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/INPUT' + unique + '.npz')
        send('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/INPUT' + unique + '.npz')
    
    try:
        # sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/DONE')
        send('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/DONE')
    except FileNotFoundError:
        # It gets deleted too fast for sftp to check!
        pass
    
    if avoid_hands:
        try:
            # sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/DONE')
            send('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/DONE')
        except FileNotFoundError:
            # It gets deleted too fast for sftp to check!
            pass
        
    # Wait for robot grasp to come through
    for _ in range(100):
        # sftp.get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/predictions_INPUT' +  unique + '.npz', 'tmp/grasp_res' + unique + '.npz')
        succ = get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/predictions_INPUT' + unique + '.npz', 'tmp/grasp_res.npz')
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
            # sftp.get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/hand_predictions_INPUT' +  unique + '.npz', 'tmp/grasp_hand_res' + unique + '.npz')
            succ = get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/hand_predictions_INPUT' +  unique + '.npz', 'tmp/grasp_hand_res.npz')
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
    except TimeoutError:
        return False
    
def get(src, dest):
    try:
        command = 'scp -i secrets/id_ed25519 corallab-s1@10.164.8.169:' + src + ' ' + dest
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=2)
        return result.returncode == 0
    except TimeoutError:
        return False

def get_glip(prompt, image):

    print("Requesting GLIP result")

    from PIL import Image
    import time

    imgname = 'glipimage' + str(int(100 * time.time())) + '-' + str(prompt.replace(' ', '_'))
    Image.fromarray(image).save('tmp/' + imgname + '.png')
    # print('pre put', imgname)
    # sftp.put('tmp/' + imgname + '.png', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/' + imgname + '.png',
    #          confirm=False)
    send('tmp/' + imgname + '.png', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/' + imgname + '.png')
    send('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/DONE')

    # print('post put')
    # try:
    #     sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/DONE')
    # except FileNotFoundError:
    #     # It gets deleted too fast for sftp to check!
    #     pass

    

    # # Wait for file to appear
    # for _ in range(10):
    #     try:
    #         sftp.stat(REMOTE_PATH)
            
    #         break
    #     except IOError:
    #         pass
    #     time.sleep(0.2)
    # else:
    #     raise Exception('Did not get the file in 2s')

    # # Get the file after short delay
    # sftp.get(REMOTE_PATH, 'tmp/dest.npz')
    
    REMOTE_PATH= f'/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/outs/' + imgname + '.npz'

    for _ in range(20):
        succ = get(REMOTE_PATH,' tmp/dest.npz')

        if succ: break

        time.sleep(0.2)

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