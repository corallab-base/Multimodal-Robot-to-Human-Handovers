
# Launch the thing
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
    print('SSH connect DONE')


def get_grasp(rgb, depth, mask, avoid_hands):
    '''
    Find a grasp in mask, avoid hand candidates if true
    '''
    print("Requesting HUMAN and ROBOT Grasp position")

    unique = str(int(100 * time.time()))

    np.savez('tmp/grasp' + unique + '.npz', depth=depth, rgb=rgb, segmap=mask, K=[911.445649104, 0, 641.169, 0, 891.51236121, 352.77, 0, 0, 1])
    
    # Save this for robot grasp gen
    sftp.put('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/INPUT' + unique + '.npz')
    
    if avoid_hands:
        # Save this for human grasp gen
        sftp.put('tmp/grasp' + unique + '.npz', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/INPUT' + unique + '.npz')
    
    try:
        sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs/DONE')
    except FileNotFoundError:
        # It gets deleted too fast for sftp to check!
        pass
    
    if avoid_hands:
        try:
            sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/inputs_hand/DONE')
        except FileNotFoundError:
            # It gets deleted too fast for sftp to check!
            pass

    # Wait for robot grasp to come through
    for _ in range(100):
        try:
            sftp.get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/predictions_INPUT' +  unique + '.npz', 'tmp/grasp_res' + unique + '.npz')
            break
        except:
            pass
        time.sleep(0.2)
    else:
        raise Exception('Did not get ROBOT GRASP result in 20s')
    print("    Got robot grasp position")
    
    data = np.load('tmp/grasp_res' + unique + '.npz', allow_pickle=True)
    pc_full, pred_grasps_cam, scores, contact_pts, pc_colors = \
        data['pc_full'], data['pred_grasps_cam'], data['scores'], data['contact_pts'], data['pc_colors']
    
    if avoid_hands:
        # Wait for human grasp to come through
        for _ in range(100):
            try:
                sftp.get('/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/CoGrasp/results/hand_predictions_INPUT' +  unique + '.npz', 'tmp/grasp_hand_res' + unique + '.npz')
                break
            except:
                pass
            time.sleep(0.2)
        else:
            raise Exception('Did not get the HUMAN HAND result in 20s')
    
        print("    Got human hand position")

        dataa = np.load('tmp/grasp_hand_res' + unique + '.npz', allow_pickle=True)
        hand_pcs, hand_cols = \
            dataa['hand_pcs'], dataa['hand_cols']
    else:
        hand_pcs, hand_cols = None, None

    return pc_full, pred_grasps_cam, scores, contact_pts, pc_colors, hand_pcs, hand_cols
    
def get_glip(prompt, image):

    print("Requesting GLIP result")

    from PIL import Image
    import time

    imgname = 'glipimage' + str(int(100 * time.time())) + '-' + str(prompt)
    Image.fromarray(image).save('tmp/' + imgname + '.png')
    print('pre put')
    sftp.put('tmp/' + imgname + '.png', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/' + imgname + '.png')
    print('post put')
    try:
        sftp.put('tmp/DONE', '/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/ins/DONE')
    except FileNotFoundError:
        # It gets deleted too fast for sftp to check!
        pass

    print('Here')
    
    REMOTE_PATH= f'/media/corallab-s1/2tbhdd1/Xuyang/part-segmentation/main/outs/' + imgname + '.npz'

    # Wait for file to appear
    for _ in range(10):
        try:
            sftp.stat(REMOTE_PATH)
            
            break
        except IOError:
            pass
        time.sleep(0.2)
    else:
        raise Exception('Did not get the file in 2s')

    # Get the file after short delay
    sftp.get(REMOTE_PATH, 'tmp/dest.npz')

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