import cv2 as cv
import numpy as np
import json
import glob
import os

# Method that draws a 3d coordinate lines into the image.
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print(corner, tuple(imgpts[0].ravel()))
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Open the camera calibration JSON file.
with open("distortion_matrix.json") as json_file:
	data = json.load(json_file)
	mtx = np.array(data["camera_matrix"])
	dist = np.array(data["distortion"])
	
# Open the camera images in current directory with glob.
cap = cv.VideoCapture(0)
	
# Use OpenCV to undistort an image from the camera, and then use solvePNP to draw a XYZ plane onto the chessboard. 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((8*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:8].T.reshape(-1,2)
# Fix units.
#objp *= 23
#print(objp)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
while(1):
	# Read images with OpenCV.
	ret, img = cap.read()
	
	# Scale the camera matrix.
	h, w = img.shape[:2]
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	
	# Undistort the image using the provided camera matrix values.
	dst = cv.undistort(img, mtx, dist, None, newcameramtx)
	
	gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
	ret, corners = cv.findChessboardCorners(gray, (4,8), None)

	if ret == True:
		corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

		# Find the rotation and translation vectors.
		_, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)
		
		# Convert the rotation matrix from the solvePNP function to a rotation vector, or vise versa. 
		rmtx, _ = cv.Rodrigues(rvecs)
		print("\nRMTX: " + str(rmtx))
		
		# Calculate the pitch, roll, yaw angles of the object.
		angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmtx)
		
		# Get the camera x, y, z translation. 
		location = -np.matrix(rmtx).T * np.matrix(tvecs)

		# Project 3D points to image plane.
		imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

		# Print object world position.
		print("RVECS: " + str(rvecs) + " \nTVECS: " + str(tvecs) + "\n")
		print("Object Orientation: \n" + str(angles) + "\n")
		print ("Object Translation: " + str(location) + "\n")

		# Draw into the image.
		img = draw(img, corners2, imgpts)

	if cv.waitKey(1) & 0xFF == ord('q'):
		cap.release()
		cv.destroyAllWindows()
		break
	
	cv.imshow("Undistorted and SolvePNP Tracked", img)
	#cv.imwrite(os.path.join("/home/ccowen/Desktop/Camera Calibration/Undistorted Images/", "image" + str(x) + ".jpg", dst)

cv.destroyAllWindows()
