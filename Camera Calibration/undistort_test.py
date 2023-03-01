import cv2
import numpy as np
import json
import os
import os


def undistort(args):
	# Open the camera calibration JSON file.
	# Open calibration for normal camera.
	if not args.fisheye:
		with open(args.camera_params_json[0]) as json_file:
			data = json.load(json_file)
			mtx = np.array(data["camera_matrix"])
			dist = np.array(data["distortion"])
	else:
		with open(args.camera_params_json[0]) as json_file:
			data = json.load(json_file)
			K = np.array(data["K_matrix"])
			D = np.array(data["D_matrix"])
			DIM = np.array(data["DIM"])

		
	# Open the camera images in current directory with glob.
	cap = cv2.VideoCapture(int(args.camera_path[0]))
		
	# Use OpenCV to undistort an image from the camera. 
	x = 0
	while True:
		# Read images with OpenCV.
		#img = cv.imread(fname)
		ret, img = cap.read()

		# Scale the camera matrix.
		h, w = img.shape[:2]

		# Check if the given camera calibration is for a fisheye lens.
		if not args.fisheye:
			# Normal Camera Matrix.
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
			# Undistort the image using the provided camera matrix values.
			dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		else:
			# Fisheye Camera Matrix.
			map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
			dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		
		if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
			cap.release()
			cv22.destroyAllWindows()
			break
		
		cv2.imshow("Undistorted", dst)
		cv2.imwrite(os.path.join("Undistorted Images/", "image" + str(x) + ".jpg"), dst)
		x += 1


if __name__ == '__main__':
	import argparse
	import sys

	# Create arguments.
	parser = argparse.ArgumentParser(description='Undistort Utility')
	parser.add_argument('--camera-params-json', '-p', nargs=1, default="output_params.json", help='The camera calibration matrix from the calibration utility.')
	parser.add_argument('--fisheye', '-f', type=bool, default=False, help='If the camera calibration is for a fisheye lens.')
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
