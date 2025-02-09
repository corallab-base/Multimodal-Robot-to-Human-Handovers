
# First import library
import pyrealsense2.pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import torchvision.transforms as transforms

CAMERA_W = 1280
CAMERA_H = 720
# CAMERA_W = 640
# CAMERA_H = 480
CAMERA_FPS = 15

class RSCapture:
    def __init__(self, serial_number=None, path = None, use_meters=False, preset='Default', dont_set_depth_preset=False, exposure=None):

        # List all cameras
        ctx = rs.context()
        for device in ctx.devices:
            this_serial_number = device.get_info(rs.camera_info.serial_number)
            this_name = device.get_info(rs.camera_info.name)
                    
        # Create pipeline
        self.pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()
        self.config = config

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        if path is not None:
            print("Reading from bag", path)
            rs.config.enable_device_from_file(config, path)
        else:
            print("Reading from live Realsense camera", serial_number)

        # Configure the pipeline to stream the depth stream
        # Change this parameter according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, CAMERA_W, CAMERA_H, rs.format.z16, CAMERA_FPS)
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.rgb8, CAMERA_FPS)

        if serial_number:
            config.enable_device(serial_number)

        # Start streaming from file
        profile = self.pipeline.start(self.config)

        print(f"    Resolution is: {CAMERA_W}x{CAMERA_H} @{CAMERA_FPS} fps")
        
        depth_sensor = profile.get_device().first_depth_sensor()

        if exposure is None:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, exposure)

        if use_meters:
            depth_sensor.set_option(rs.option.depth_units, 0.0001)
        self.depth_scale = depth_sensor.get_depth_scale()
        print("    Depth Scale is: " , self.depth_scale)
        
        # if not dont_set_depth_preset:
        #     preset_range = depth_sensor.get_option_range(rs.option.visual_preset)

        #     preset_found = False
        #     for i in range(int(preset_range.max)):
        #         visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
        #         if visual_preset == preset:
        #             depth_sensor.set_option(rs.option.visual_preset, i)
        #             preset_found = True
        #             print("   ", visual_preset, "preset selected")

        #     if not preset_found:
        #         print("   Desired preset", preset, 'not found from', preset_range)

        

        # depth_sensor.set_option(rs.option.confidence_threshold, 1) # 1 -3
        # depth_sensor.set_option(rs.option.noise_filtering, 6)

        self.align = rs.align(rs.stream.color)
        # self.pc = rs.pointcloud()
        
        if path is not None:
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)

        # Create colorizer object
        colorizer = rs.colorizer()

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_frames(viz=False)
    
    def get_frames(self, rotate=False, viz=False):
        
        # # Create opencv window to render image in
        # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        
        try:
            # Get frameset of depth
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)

            # playback.pause()

            # Get color and depth, aligned to depth
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            # self.pose_of_bb((0, 100, 0, 100), depth_frame)

            # Colorize depth frame to jet colormap
            # depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Render image in opencv window
            if viz:
                cv2.imshow("Depth Stream", depth_image)
                cv2.imshow("Color Stream", color_image)
                # print(depth_color_image.min(), depth_color_image.max())
                cv2.waitKey(1)

            # playback.resume()

            def rotate180(image):
                # Flip y only. Do nothing to X so that it is mirrored
                return image[::-1, ::-1 ,...]

            if rotate:
                return rotate180(color_image), rotate180(depth_image), depth_frame, color_frame
            else:
                return color_image, depth_image, depth_frame, color_frame
        
        except Exception as err:
            print("Exception getting frame:", err)
            return None
        
    def pose_of_bb(self, bbox, depth):
        right, left, high, low = bbox

        left = int(left)
        right = int(right)
        low = int(low)
        high = int(high)

        print('Bounding box of face:', bbox)

        # Get image dims
        w = rs.video_frame(depth).width
        h = rs.video_frame(depth).height

        # Convert to 3D verts
        pointcloud = self.pc.calculate(depth)
        verts = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(h, w, 3)
        
        # Cut out the points we want
        roi = verts[low:high, left:right, :]

        # Filter nearer 50% percent
        median = np.nanmedian(roi[:, :, 2]) # sort by z dist
        near = roi[roi[:, :, 2] < median]

        # mean over x, y, z independently
        avg_depth = np.nanmean(near.reshape(-1, 3),  axis=0) 
        print('Avg coordinate in bounding box:', avg_depth)

        return avg_depth

class WebCamCapture:
    def __init__(self) -> None:
        self.vc = cv2.VideoCapture(0)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.vc.read()[1]
    
    def pose_of_bb(self, bbox, depth):
        right, left, high, low = bbox

        left = int(left)
        right = int(right)
        low = int(low)
        high = int(high)

        print('Bounding box of face:', bbox)

        # Get image dims
        w = rs.video_frame(depth).width
        h = rs.video_frame(depth).height

        # Convert to 3D verts
        pointcloud = self.pc.calculate(depth)
        verts = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(h, w, 3)
        
        # Cut out the points we want
        roi = verts[low:high, left:right, :]

        # Filter nearer 50% percent
        median = np.nanmedian(roi[:, :, 2]) # sort by z dist
        near = roi[roi[:, :, 2] < median]

        # mean over x, y, z independently
        avg_depth = np.nanmean(near.reshape(-1, 3),  axis=0) 
        print('Avg coordinate in bounding box:', avg_depth)

        return avg_depth


if __name__ == "__main__":
    cap = RSCapture()
    # cap = WebCamCapture()

    for rgb, depth, _, _ in cap:
        cv2.imshow('gazes', rgb)
        cv2.imshow('depths', depth)
        cv2.waitKey(1)
