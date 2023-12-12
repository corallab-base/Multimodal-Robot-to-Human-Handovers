import cv2

import pykinect_azure as pykinect
from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_HAND_RIGHT
from sympy.physics.vector import ReferenceFrame, express, Vector
import math
import quaternion
import numpy as np

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries(track_body=True)

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
#print(device_config)

# Start device
device = pykinect.start_device(config=device_config)

# Start body tracker
bodyTracker = pykinect.start_body_tracker()

cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)

# Set up frame

origin = ReferenceFrame('origin')
kinect = ReferenceFrame('kinect')

kinect.orient(origin, 'Axis', [math.radians(45), origin.z])

def get_hand_pose():
	# Get capture
	capture = device.update()

	# Get body tracker frame
	body_frame = bodyTracker.update()

	# Get the color depth image from the capture
	ret_depth, depth_color_image = capture.get_colored_depth_image()

	# Get the colored body segmentation
	ret_color, body_image_color = body_frame.get_segmentation_image()

	if not ret_depth or not ret_color:
		return
		
	# Combine both images
	combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

	# Draw the skeletons
	combined_image = body_frame.draw_bodies(combined_image)

	target_pos_in_origin = None
	
	if body_frame.get_num_bodies() > 0:
		wrist = body_frame.get_body_skeleton().joints[K4ABT_JOINT_WRIST_RIGHT]
		hand = body_frame.get_body_skeleton().joints[K4ABT_JOINT_WRIST_RIGHT]

		orientation = np.quaternion(hand.orientation.wxyz.w, hand.orientation.wxyz.x, hand.orientation.wxyz.y, hand.orientation.wxyz.z)
		
		# Flip some axes to match our coords
		position = np.asarray([hand.position.xyz.z, -hand.position.xyz.x, -hand.position.xyz.y])

		# Also convert mm to m
		position /= 1000.

		print(f'hand pos: {position} / orient: {orientation}\n\n')

		# Project the hand orientation to get the empty space in front of hand
		# Say at a distance of 10 cm.
		hand_orientation_vec = quaternion.as_rotation_matrix(orientation).dot([1, 0, 0])

		target_pos = position + 0.1 * hand_orientation_vec

		# Define the target pose to the Kinect frame
		target_pos_in_kinect = target_pos[0] * kinect.x + target_pos[1] * kinect.y + target_pos[2] * kinect.z

		# Express the pose in the Origin frame
		target_pos_in_origin = express(target_pos_in_kinect, origin)

		target_pos_in_origin = [target_pos_in_origin.args[0][0][0], target_pos_in_origin.args[0][0][1], target_pos_in_origin.args[0][0][2]]
		
		# Translate
		target_pos_in_origin[0] += -0.48
		target_pos_in_origin[1] += 0.46
		target_pos_in_origin[2] += 0.34


	# Overlay body segmentation on depth image
	cv2.imshow('Depth image with skeleton',combined_image)

	# Press q key to stop
	if cv2.waitKey(1) == ord('q'):  
		return None
	
	return target_pos_in_origin
	
if __name__ == "__main__":
	while True:
		print(get_hand_pose())