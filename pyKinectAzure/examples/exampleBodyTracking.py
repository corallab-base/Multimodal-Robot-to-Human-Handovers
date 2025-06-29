import cv2

import pykinect_azure as pykinect
from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_HAND_RIGHT

if __name__ == "__main__":

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
	while True:

		# Get capture
		capture = device.update()

		# Get body tracker frame
		body_frame = bodyTracker.update()

		# Get the color depth image from the capture
		ret_depth, depth_color_image = capture.get_colored_depth_image()

		# Get the colored body segmentation
		ret_color, body_image_color = body_frame.get_segmentation_image()

		if not ret_depth or not ret_color:
			continue
			
		# Combine both images
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

		# Draw the skeletons
		combined_image = body_frame.draw_bodies(combined_image)
		
		if body_frame.get_num_bodies() > 0:
			wrist = body_frame.get_body_skeleton().joints[K4ABT_JOINT_WRIST_RIGHT].position
			hand = body_frame.get_body_skeleton().joints[K4ABT_JOINT_WRIST_RIGHT].position

			print(f'wrist pos {wrist.xyz} \n hand pos {hand.xyz} \n\n')

		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break