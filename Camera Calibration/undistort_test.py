import cv2 as cv
import numpy as np
import json
import os
import os


def undistort(args):
	# Open the camera calibration JSON file.
	with open(args.camera_params_json[0]) as json_file:
		data = json.load(json_file)
		mtx = np.array(data["camera_matrix"])
		dist = np.array(data["distortion"])
		
	# Open the camera images in current directory with glob.
	cap = cv.VideoCapture(int(args.camera_path[0]))
		
	# Use OpenCV to undistort an image from the camera. 
	x = 0
	while True:
		# Read images with OpenCV.
		#img = cv.imread(fname)
		ret, img = cap.read()
		
		# Scale the camera matrix.
		h, w = img.shape[:2]
		newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		
		# Undistort the image using the provided camera matrix values.
		dst = cv.undistort(img, mtx, dist, None, newcameramtx)
		
		if cv.waitKey(1) & 0xFF == ord('q') or ret==False :
			cap.release()
			cv2.destroyAllWindows()
			break
		
		cv.imshow("Undistorted", dst)
		cv.imwrite(os.path.join("Undistorted Images/", "image" + str(x) + ".jpg"), dst)
		x += 1


if __name__ == '__main__':
	import argparse
	import sys

	# Create arguments.
	parser = argparse.ArgumentParser(description='Undistort Utility')
	parser.add_argument('--camera-params-json', '-p', nargs=1, default="output_params.json", help='The camera calibration matrix from the calibration utility.')
	parser.add_argument('--camera-path', '-i', nargs=1, default=0, help='The index or path of the camera, video file to undistort.')
	args = parser.parse_args()

	if sys.platform == "win32":
		# windows does not expand the "*" files on the command line
		#  so we have to do it.
		import glob

		infiles = []
		for f in args.input_files:
			infiles.extend(glob.glob(f))
			args.input_files = infiles

	# Undistort images.
	undistort(args)
