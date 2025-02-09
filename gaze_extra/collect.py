import os
import sys
from PIL import Image
import numpy as np
import cv2

from gaze_utils.realsense import RSCapture  # For displaying images in a window

v_reader = RSCapture(dont_set_depth_preset=True, exposure=10)

def capture_and_save_with_user_interaction():
    # Define the directory to save images
    directory = "gaze_extra/images"
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Get the list of existing files in the directory
    existing_files = [f for f in os.listdir(directory) if f.startswith("image_") and f.endswith(".png")]
    existing_files.sort()  # Sort files to find the next increment

    # Determine the next image number
    if existing_files:
        # Extract the number from the last file
        last_file = existing_files[-1]
        last_num = int(last_file.split("_")[1].split(".")[0])
        next_num = last_num + 1
    else:
        next_num = 0

    while True:
        frame = v_reader.get_frames()
        image_rgb, depth_image, depth_frame, color_frame = frame

        # Convert the NumPy array to a PIL Image (ensure dtype is uint8)
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)  # Normalize if necessary

        pil_image = Image.fromarray(image_rgb, mode="RGB")

        # Display the image using OpenCV (Pillow doesn't have a simple display feature)
        cv2.imshow("Captured Image", np.array(pil_image))

        # Ask user whether to save or retry
        print("Hit 'y' to save this image, or 'n' to try again.")

        key = cv2.waitKey(0)  # Wait for user input
        if key == ord('y'):  # User pressed 'y' (save and continue)
            file_name = f"image_{next_num:03d}.png"
            file_path = os.path.join(directory, file_name)

            # Save the image as PNG
            pil_image.save(file_path)
            print(f"Image saved to: {file_path}")
            next_num += 1
            break
        elif key == ord('n'):  # User pressed 'n' (retry)
            print("Retrying...")
            continue
        elif key == 27:  # Escape key to exit
            print("Exiting without saving.")
            sys.exit(0)

    # Close the OpenCV window
    cv2.destroyAllWindows()

while True:
    capture_and_save_with_user_interaction()