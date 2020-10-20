# realsense camera test file

import numpy as np 
import os
import time
import cv2
import pyrealsense2 as rs 

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color

        key = input("Enter any key to continue while q to quit")
        if key != 'q':
            pass
        else:
            break

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        curr_time = str(time.time())
        images = np.hstack((color_image, depth_colormap))
        color_name = 'color_' + curr_time + '.jpg'
        depth_name = 'depth_' + curr_time + '.jpg'

        color_path = os.path.join('../datas/centrifuges/color/', color_name)
        depth_path = os.path.join('../datas/centrifuges/depth/', depth_name)        
        cv2.imwrite(color_path, color_image)
        cv2.imwrite(depth_path, depth_image)

        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
        #cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()