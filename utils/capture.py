import numpy as np 
import cv2
import pyrealsense2 as rs

def capture_rs():
	"""
	利用realsense2摄像图片拍摄头, 拍摄图片
	"""
	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	# Start streaming
	pipeline.start(config)

	frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return -1

    # Convert images to numpy arrays
    depth_image = np.asarray(depth_frame.get_data())
    color_image = np.asarray(color_frame.get_data())

    pipeline.stop()

    return [color_image, depth_image]

def main():
	pass


if __name__ == '__main__':
	main()