import argparse
import multiprocessing
import threading
import os
import time

import cv2
import torch


from gaze_utils.realsense import RSCapture

from gaze_utils.sshlib import connect as connectSSH

print('Importing Headpose Model...')
from gaze_utils.headpose import infer, free_model

from mover import moveit, init as initArm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def wait_for_file(filename):
    while True:
        file_exists = os.path.exists(filename)
        if file_exists: break
        time.sleep(0.2)
        
def gaze_track(stop_gaze_tracking, display_image):
    v_reader = RSCapture(serial_number='123122060050', dont_set_depth_preset=True)

    heatmap = None

    for frame in v_reader:
        color_image, depth_image, depth_frame, color_frame = frame

        # Run headpose tracking
        heatmap = infer(color_image, display_image, viz=True)

        with stop_gaze_tracking.get_lock():  # Synchronize access to the shared value
            if stop_gaze_tracking.value:
                free_model()
                print(f"Saving heatmap to heatmap.pth")
                torch.save(heatmap, 'heatmap.pth')

                cv2.destroyAllWindows()
                return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Process program parameters.")

    parser.add_argument("--real", default=True, type=str2bool,
                        help="Whether the program is executing for real, otherwise use saved data from last run.")
    parser.add_argument("--gaze", default=True, type=str2bool,
                        help="Whether to run gaze tracking, otherwise use saved data from last run.")
    parser.add_argument("--prompt", default=None, type=str, 
                        help="If given, use in place of audio recording.")

    args = parser.parse_args()
    
    # connectSSH()

    stop_gaze_tracking = multiprocessing.Value('b', False)

    def stop():
        with stop_gaze_tracking.get_lock(): 
            stop_gaze_tracking.value = True

    multiprocessing.Process(target=initArm,
                            args=(args.real,)
                            ).start()

    wait_for_file('capture_pointcloud/front_rgb')
    front_rgb = torch.load('capture_pointcloud/front_rgb')

    if args.gaze:
        threading.Thread(target=gaze_track, 
                         args=(stop_gaze_tracking, front_rgb)
                         ).start()
        
    print('Importing SpaCy Model...')
    from gaze_utils.text_extraction import record_and_parse

    objname, partname, target_holder = record_and_parse(args.prompt, recording_done_func=stop)

    multiprocessing.Process(target=moveit,
                                kwargs={'real_life': args.real, 
                                        'objname': objname, 'partname': partname, 'target_holder': target_holder}
                            ).start()