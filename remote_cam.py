
'''
For calibrating camera_tweaks() in mover.py and the K instrinsics in constants.py.

Place a blue object below the midline and run this script. The arm will move to point to the object
(or where it thinks it is). By repeating this for multiple positions (left, right, close, far) ofthe
blue ball, you can get an idea for what is off in the parameters.
'''


# Imports
import cv2
import gzip
import numpy as np
import socket

from gaze_utils.realsense import RSCapture

# Connection constants
HOST = "192.168.1.149"
PORT = 3009

client_socket = None

# # Gets image using the direct connection
# def get_image_depth():
#     wrist_cam = RSCapture(serial_number='044122070299', use_meters=False, preset='Default')
#     color_image, depth_image, depth_frame, color_frame = wrist_cam.get_frames()
#     return color_image, depth_image

# Gets the image from the remote camera
# def get_image():
#     global client_socket

#     print('Connecting to remote camera...')
#     client_socket = socket.socket()
#     client_socket.connect((HOST, PORT))
#     chunk, data = None, None

#     # Getting data from the server
#     data = client_socket.recv(4)
#     data_len = int.from_bytes(data, byteorder="big")
#     data = b''
#     while len(data) < data_len - 2048:
#         data += client_socket.recv(2048)
#     data += client_socket.recv(data_len - len(data))

#     # Decompressing the image data
#     data = gzip.decompress(data)

#     # Reshape the received image data into a numpy array
#     print("data:", len(data))
#     image = np.frombuffer(data, dtype=np.uint8).reshape((720, 1280, 3))
#     return image


# Gets the depth from the remote camera
# def get_depth():
#     global client_socket

#     # Getting data from the server
#     data = b''
#     while True:
#         chunk = client_socket.recv(2048)
#         if not chunk:
#             break
#         data += chunk

#     # Decompressing the image data
#     data = gzip.decompress(data)
    
#     # Reshaping the received depth data into a numpy array
#     depth = np.frombuffer(data, dtype=np.float32).reshape((720, 1280))
#     return depth


# # Imports
import sys
sys.path.append('/home/corallab/librealsense/build/Release')

import cv2
import gzip
import numpy as np
import pyrealsense2 as rs
import socket


# Camera Constants
CAMERA_W = 1280
CAMERA_H = 720
CAMERA_FPS = 30

# Server Constants
HOST = '192.168.1.149'
PORT = 3009


# RS Capture Class
class Deeeeez:
    def __init__(self, serial_number=None, use_meters=False, preset='Default', dont_set_depth_preset=False):
        # Create pipeline
        self.pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()
        self.config = config

        # Configure the pipeline to stream the color and depth stream
        config.enable_stream(rs.stream.depth, CAMERA_W, CAMERA_H, rs.format.z16, CAMERA_FPS)
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.rgb8, CAMERA_FPS)

        if serial_number:
            config.enable_device(serial_number)

        # Start streaming from file
        profile = self.pipeline.start(self.config)
        
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto exposure
        color_sensor.set_option(rs.option.exposure, 700)  # Set manual exposure

        depth_sensor = profile.get_device().first_depth_sensor()
        if use_meters:
            depth_sensor.set_option(rs.option.depth_units, 0.0001)
        self.depth_scale = depth_sensor.get_depth_scale()


        print(f"\tResolution is: {CAMERA_W}x{CAMERA_H} @ {CAMERA_FPS} fps\n\tDepth Scale is: " , self.depth_scale)
        
        
        if not dont_set_depth_preset:
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)

            preset_found = False
            for i in range(int(preset_range.max)):
                visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                if visual_preset == preset:
                    depth_sensor.set_option(rs.option.visual_preset, i)
                    preset_found = True
                    print(f'\t{visual_preset} preset selected')

            if not preset_found:
                print(f'\tDesired preset {preset} not found from {preset_range}.')
        
        self.align = rs.align(rs.stream.color)

        # Create colorizer object
        colorizer = rs.colorizer()


    # Gets the next set of frames
    def get_frames(self, rotate=False, viz=False):
        try:
            # Get frameset of depth
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)

            # Get color and depth, aligned to depth
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            # Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Render image in opencv window
            if viz:
                cv2.imshow("Depth Stream", depth_image)
                cv2.imshow("Color Stream", color_image)
                cv2.waitKey(1)

            def rotate180(image):
                # Flip y only. Do nothing to X so that it is mirrored
                return image[::-1, : ,...]

            if rotate:
                return rotate180(color_image), rotate180(depth_image), depth_frame, color_frame
            else:
                return color_image, depth_image

        except Exception as err:
            print("Exception getting frame:", err)
            return None

depth = None
wrist_cam = Deeeeez(serial_number='044122070299', use_meters=False, preset='Default')
def get_image():

    global depth
    rgb, depth = wrist_cam.get_frames()
    return rgb

def get_depth():
    return depth

if __name__ == '__main__':
    print('Fetching remote color image...')
    a = get_image()
    cv2.imshow('color', a)
    cv2.waitKey(1000)

    print('Fetching remote depth image...')
    b = get_depth()
    normalized_depth = np.clip(b, 0, 65535) # Assuming 16-bit image, change as necessary
    normalized_depth = (normalized_depth / 65535) * 255
    normalized_depth = normalized_depth.astype(np.uint8)
    cv2.imshow('depth', b)
    cv2.waitKey(0)