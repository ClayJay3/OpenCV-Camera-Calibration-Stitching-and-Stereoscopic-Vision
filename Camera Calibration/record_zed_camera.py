import cv2
import os
import pyzed.sl as sl
import numpy as np
import argparse


def record_zed(args):
	# Print debub
	print("Recording video.mp4 from zed camera... (PRESS Q TO STOP)")

	# Create a ZED camera object
	zed = sl.Camera()

	# Set configuration parameters
	init_params = sl.InitParameters()
	init_params.camera_resolution = sl.RESOLUTION.HD720
	init_params.camera_fps = 30

	# Open the camera
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS:
		exit(-1)

	# Get ZED data.
	image_size = zed.get_camera_information().camera_resolution

	# Create custom mat for zed image.
	image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

	# Create opencv video writer for exporting video.
	video = cv2.VideoWriter('zed_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (image_size.width, image_size.height), isColor=True)

	# Loop forever.
	while(True):
		if zed.grab() == sl.ERROR_CODE.SUCCESS:
			# Check if we should grab images from the left or right camera.
			if (args.lefteye):
				# A new image is available if grab() returns SUCCESS
				zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Retrieve the left image
			else:
				# A new image is available if grab() returns SUCCESS
				zed.retrieve_image(image_zed, sl.VIEW.RIGHT) # Retrieve the left image

			# Get camera frame from zed.
			img = image_zed.get_data()
			img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

			# Write frame from zed.
			video.write(img.copy())

			# Show frame.
			cv2.imshow("Zed Frame", img)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				# When everything done, release the video capture and video write objects.
				video.release()
				zed.close()
				cv2.destroyAllWindows()
				break

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
	# Create arguments.
	parser = argparse.ArgumentParser(description='Video Split Untility')
	parser.add_argument('--lefteye', '-l', type=boolean_string, default=False, help='Whether or not to use the left or right camera.')

	args = parser.parse_args()
	
	record_zed(args)