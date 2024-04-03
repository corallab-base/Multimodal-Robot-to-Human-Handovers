
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


# Connection constants
HOST = "192.168.1.149"
PORT = 3008


print('Connecting to remote camera...')
client_socket = socket.socket()
client_socket.connect((HOST, PORT))
chunk, data = None, None

# Gets the image from the remote camera
def get_image():
    # Getting data from the server
    data = client_socket.recv(4)
    data_len = int.from_bytes(data, byteorder="big")
    data = b''
    while len(data) < data_len - 2048:
        data += client_socket.recv(2048)
    data += client_socket.recv(data_len - len(data))

    # Decompressing the image data
    data = gzip.decompress(data)

    # Reshape the received image data into a numpy array
    image = np.frombuffer(data, dtype=np.uint8).reshape((720, 1280, 3))
    return image


# Gets the depth from the remote camera
def get_depth():
    # Getting data from the server
    data = b''
    while True:
        chunk = client_socket.recv(2048)
        if not chunk:
            break
        data += chunk

    # Decompressing the image data
    data = gzip.decompress(data)
    
    # Reshaping the received depth data into a numpy array
    depth = np.frombuffer(data, dtype=np.float32).reshape((720, 1280))
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