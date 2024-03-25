import socket
import sys
import cv2
import numpy as np
from PIL import Image

print('Connecting to remote camera...')
client_socket = socket.socket()
client_socket.connect(("192.168.1.149", 3003))

# with open('file.npy', 'wb') as f:
#     l = s.recv(1024)
#     while len(l) > 0:
#         f.write(l)
#         l = s.recv(1024)

# recv_file = np.load('file.npy', allow_pickle=True)
# Image.fromarray(recv_file[:, :, :3].astype(np.uint8)).show()
# s.close()

COLOR_IMAGE_BYTES = 1280 * 720 * 3

chunk, image_data = None, None

def get_image():
    global chunk, image_data

    print('Fetching remote color image...')
    # Getting data from the server
    image_data = b''
    chunk = client_socket.recv(1024)
    while len(chunk) > 0:
        image_data += chunk
        chunk = client_socket.recv(1024)

        if len(image_data) >= COLOR_IMAGE_BYTES:
            break

    # Decode the received image data into a numpy array
    if len(image_data) == 0:
        raise Exception('Got nothing from camera')
    image = np.frombuffer(image_data[:COLOR_IMAGE_BYTES], dtype=np.uint8).reshape((720, 1280, 3))
    return image

def get_depth():
    '''
    call after get_color
    '''
    global chunk, image_data

    print('Fetching remote depth image...')

    while len(chunk) > 0:
        image_data += chunk
        chunk = client_socket.recv(1024)

    # Decode the received image data into a numpy array
    if len(image_data) == 0:
        raise Exception('Got nothing from camera')
    image = np.frombuffer(image_data[COLOR_IMAGE_BYTES:], dtype=np.float32).reshape((720, 1280))
    return image


if __name__ == '__main__':
    a = get_image()
    cv2.imshow('color', a)
    cv2.waitKey(1000)

    b = get_depth()
    normalized_depth = np.clip(b, 0, 65535) # Assuming 16-bit image, change as necessary
    normalized_depth = (normalized_depth / 65535) * 255
    normalized_depth = normalized_depth.astype(np.uint8)
    cv2.imshow('depth', b)
    cv2.waitKey(0)