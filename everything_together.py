

def get_grasp(ply_file_path, ply_file_dest, ply_part_file=None):
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    k = paramiko.Ed25519Key.from_private_key_file('secrets/id_ed25519')
    ssh.connect(hostname='10.164.8.169', username='corallab-s1', pkey=k)

    pycmd = 'predict.py -r -f test_data/input.ply -o test_data/out.npy -v True'

    sftp = ssh.open_sftp()
    sftp.put(ply_file_path, '/media/corallab-s1/2tbhdd/Xuyang/part-segmentation/CoGrasp/test_data/input.ply')

    if ply_part_file is not None:
        sftp.put(ply_part_file, '/media/corallab-s1/2tbhdd/Xuyang/part-segmentation/CoGrasp/test_data/part.ply')
        pycmd += '-p test_data/part.ply'

    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
        '''
        cd /media/corallab-s1/2tbhdd/Xuyang
        docker stop part-segmentation-container 
        docker start part-segmentation-container 
        docker exec -w /part-segmentation/CoGrasp -t part-segmentation-container /opt/conda/envs/grasp/bin/python {pycmd}
        '''.format(pycmd=pycmd))

    while True:
        line = ssh_stdout.readline()
        if not line:
            break
        print(line, end="")

    sftp.get('/media/corallab-s1/2tbhdd/Xuyang/part-segmentation/CoGrasp/test_data/out.npy', ply_file_dest)

    sftp.close()

def get_part(ply_file_path, part_name: str, ply_file_dest):
    assert(type(part_name) is str)

    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    k = paramiko.Ed25519Key.from_private_key_file('secrets/id_ed25519')
    ssh.connect(hostname='10.164.8.169', username='corallab-s1', pkey=k)

    sftp = ssh.open_sftp()
    sftp.put(ply_file_path, '/media/corallab-s1/2tbhdd/Xuyang/part-segmentation/main/In.ply')

    part_cmd = f"import demo; demo.Infer('In.ply', ['{part_name}'], zero_shot=False, save_dir='outs')"

    cmd='''
    cd /media/corallab-s1/2tbhdd/Xuyang/
    docker stop part-segmentation-container 
    docker start part-segmentation-container 
    docker exec -w /part-segmentation/main -t part-segmentation-container /opt/conda/envs/grasp/bin/python -c \"{part_cmd}\"
    '''.format(part_cmd=part_cmd)

    print(cmd)

    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)


    while True:
        line = ssh_stdout.readline()
        if not line:
            break
        print(line, end="")

    sftp.get(f'/media/corallab-s1/2tbhdd/Xuyang/part-segmentation/main/outs/semantic_seg/{part_name}.ply', ply_file_dest)

    sftp.close()


def gaze_tracking():
    import gaze_tracking 

    gaze_tracking.main('cam', 'gaze_video.mp4', 'gaze_tracking_results.pt', 'the banana', True)

    print('Saved to last_gaze_result.pt')

def obj_find(prompt2='red apple', dest_ply='obj.ply'):
    import gaze_utils.realsense as rs
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt 
    import torch

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # tabletop_reader = rs.RSCapture(serial_number='123122060050')
    # import time
    # time.sleep(4)

    # tabletop_color, tabletop_depth, _, _ = tabletop_reader.get_frames(rotate=False, viz=False)

    print('Reading last_gaze_result.pth')

    loaded = torch.load('last_gaze_result.pth')
    tabletop_color, tabletop_depth, heatmap, prompt = loaded
    prompt = prompt2
    # tabletop_depth = cv2.imread('test_images/test_depth.jpeg')
    # tabletop_depth = tabletop_depth[:, :, 2]
    # tabletop_color = cv2.imread('test_images/test_color.jpg')

    assert(tabletop_color.shape[0] == tabletop_depth.shape[0])    
    assert(tabletop_color.shape[1] == tabletop_depth.shape[1])    
    assert(tabletop_color.shape[2] == 3)       
    assert(tabletop_color.dtype == np.uint8)
    assert(len(tabletop_depth.shape) == 2)

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tabletop_depth, alpha=0.03), cv2.COLORMAP_JET).astype(np.uint8)
    plt.figure()
    plt.imshow(tabletop_depth)
    plt.figure()
    plt.imshow(tabletop_color)
    plt.pause(0.5)

    from gaze_heatmap_to_mask import choose_object
    input_dict = choose_object(tabletop_color, tabletop_depth, heatmap, prompt, topk=10, viz=True)
    

    # Filter pc

    pc = input_dict['pc']
    rgb = input_dict['pcrgb']

    mean = np.median(pc, axis=0)
    print(mean.shape)
    dist = np.sqrt(((pc - mean)**2).sum(axis=-1))
    print(dist.shape)
    dist_mask = dist < 0.5 # one meter
    pc = pc[dist_mask]
    rgb = rgb[dist_mask]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(dest_ply, pcd)

    plt.show()

if __name__ == '__main__':
    # gaze_tracking()

    obj_find('jug', 'obj.ply')

    # get_part('obj.ply', 'handle', 'part-seg.ply')

    # get_grasp('obj.ply', 'grabber.npy')

    # from real_exp_guna import main as start_grasp

    # start_grasp(input_dict)
