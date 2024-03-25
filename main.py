import multiprocessing
import threading

import cv2
import torch

from gaze_utils.text_extraction import record_and_parse
from gaze_utils.realsense import RSCapture

from gaze_utils.sshlib import connect as connectSSH

from gaze_utils.headpose import infer, free_model

from mover import moveit
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def gaze_track(stop_gaze_tracking):
    v_reader = RSCapture(serial_number='123122060050', dont_set_depth_preset=True)

    heatmap = None

    for frame in v_reader:
        color_image, depth_image, depth_frame, color_frame = frame

        # Run headpose tracking
        heatmap = infer(color_image, viz=True)

        with stop_gaze_tracking.get_lock():  # Synchronize access to the shared value
            if stop_gaze_tracking.value:
                free_model()
                print(f"Saving heatmap to heatmap.pth")
                torch.save(heatmap, 'heatmap.pth')
                cv2.destroyAllWindows()
                return
            
connectSSH()

stop_gaze_tracking = multiprocessing.Value('b', False)
def stop():
    with stop_gaze_tracking.get_lock(): 
        stop_gaze_tracking.value = True

p = threading.Thread(target=gaze_track, 
                            args=(stop_gaze_tracking,)
                            ).start()

objname, partname, target_holder = record_and_parse("Give me the apple", recording_done_func=stop)

multiprocessing.Process(target=moveit,
                            kwargs={'real_life':True, 
                            'objname':objname, 'partname':partname, 'target_holder':target_holder}
                        ).start()