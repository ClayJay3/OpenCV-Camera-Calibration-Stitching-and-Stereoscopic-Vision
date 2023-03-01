import time
import cv2
import numpy
import os.path
import math
import argparse
import json
import logging
from pathlib import Path
import numpy as np


def is_valid_file(parser, arg):
    """
    This method is called by argparse when the user passes in a file path.
    It checks the file path and makes sure the program can open it.

    Parameters:
    -----------
        parser - The ArgumentParser object from the argparse class. Used to print errors and warnings.
        arg - The file path arg from the ArgumentParser.

    Returns:
    --------
        file - The opened file object.
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


class CameraCalibration(object):
    """Calibrates a distorted camera"""

    def __init__(self, height, width, size, camera_A_params, camera_B_params):
        # Number of inside corners in the chessboard (height and width)
        self.checkerboard_height = height
        self.checkerboard_width = width
        # Size of chessboard square in physical units
        self.square_size = size

        # Open the camera calibrations JSON file.
        with open(camera_A_params.name) as json_file:
            data = json.load(json_file)
            self.camera1_mtx = np.array(data["camera_matrix"])
            self.camera1_dist = np.array(data["distortion"])
        # Open the camera calibrations JSON file.
        with open(camera_B_params.name) as json_file:
            data = json.load(json_file)
            self.camera2_mtx = np.array(data["camera_matrix"])
            self.camera2_dist = np.array(data["distortion"])

    def calibrateCamera(self, left_images, right_images, output_dir=None):
        """
        Detects the corners of a chessboard and analyzes how the deform across an image to determine distortion and camera parameters.

        Parameters:
        -----------
            images - 
        """

        # Termination criteria. CHANGE THIS IF CALIBRATION IS NOT GOOD.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
        objp = numpy.zeros((self.checkerboard_height * self.checkerboard_width, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[0:self.checkerboard_width, 0:self.checkerboard_height].T.reshape(-1, 2)

        # calibrate coordinates to the physical size of the square
        objp *= self.square_size

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        left_imgpoints = [] # 2d points in image plane.
        right_imgpoints = [] # 2d points in image plane.

        # Loop through images and 
        frame_counter = 0
        for left_image, right_image in zip(left_images, right_images):
            # Print info.
            print(f"Processing frame: {frame_counter}")

            # Convert image to grayscale.
            left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(left_image_gray, (self.checkerboard_width, self.checkerboard_height), None)
            ret2, corners2 = cv2.findChessboardCorners(right_image_gray, (self.checkerboard_width, self.checkerboard_height), None)

            # If found, add object points, image points (after refining them).
            if ret1 and ret2:
                new_corners1 = cv2.cornerSubPix(left_image_gray, corners1, (11, 11), (-1, -1), criteria)
                new_corners2 = cv2.cornerSubPix(right_image_gray, corners2, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                detection_image1 = cv2.drawChessboardCorners(left_image, (self.checkerboard_width, self.checkerboard_height), corners1, ret1)
                detection_image2 = cv2.drawChessboardCorners(right_image, (self.checkerboard_width, self.checkerboard_height), corners2, ret2)

                # Add object points and detection points to lists.
                objpoints.append(objp)
                left_imgpoints.append(new_corners1)
                right_imgpoints.append(new_corners2)

                # Show user image detections.
                cv2.imshow("camera1 detections", detection_image1)
                cv2.imshow("camera2 detections", detection_image2)

                # Write output images to dir if enabled.
                if output_dir:
                    fullname1 = os.path.join(output_dir, os.path.basename(frame_counter), "left")
                    cv2.imwrite(fullname1, detection_image1)
                    fullname2 = os.path.join(output_dir, os.path.basename(frame_counter), "right")
                    cv2.imwrite(fullname2, detection_image2)

                cv2.waitKey(500)
            else:
                print(f"Failed to detect chessboard corners for image {frame_counter}...")

            # Increment Counter.
            frame_counter += 1

        # Close all GUI windows.
        cv2.destroyAllWindows()

        # Check if we got at least one detection.
        if not objpoints:
            print("No useful images. Quitting...")
            return None

        # Print info.
        print("Found {} useful images".format(len(objpoints)))

        # Calculate stereo calibration.
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, self.camera1_mtx, self.camera1_dist, self.camera2_mtx, self.camera2_dist, left_image_gray.shape[::-1], criteria = criteria, flags = cv2.CALIB_FIX_INTRINSIC)

        # Convert the rotation and vertical offset into remapping matrices that can be directly used to correct the stereo pair
        print("Computing Calibration")
        OPTIMIZE_ALPHA = 0.25
        (leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                self.camera1_mtx, self.camera1_dist,
                self.camera2_mtx, self.camera2_dist,
                left_image_gray.shape[::-1], R, T,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
                self.camera1_mtx, self.camera1_dist, leftRectification,
                leftProjection, left_image_gray.shape[::-1], cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                self.camera2_mtx, self.camera2_dist, rightRectification,
                rightProjection, right_image_gray.shape[::-1], cv2.CV_32FC1)

        # Print useful info and calculate camera FOV.
        print(f"Reprojection error = {ret}")
        print("Left image center = ({:.2f}, {:.2f})".format(self.camera1_mtx[0][2], self.camera1_mtx[1][2]))
        print("Right image center = ({:.2f}, {:.2f})".format(self.camera2_mtx[0][2], self.camera2_mtx[1][2]))
        fov_x = math.degrees(2.0 * math.atan(left_image_gray.shape[1] / 2.0 / self.camera1_mtx[0][0]))
        fov_y = math.degrees(2.0 * math.atan(left_image_gray.shape[0] / 2.0 / self.camera1_mtx[1][1]))
        print("Left FOV = ({:.2f}, {:.2f}) degrees".format(fov_x, fov_y))
        print("mtx =", self.camera1_mtx)
        print("dist =", self.camera1_dist)
        fov_x = math.degrees(2.0 * math.atan(right_image_gray.shape[1] / 2.0 / self.camera2_mtx[0][0]))
        fov_y = math.degrees(2.0 * math.atan(right_image_gray.shape[0] / 2.0 / self.camera2_mtx[1][1]))
        print("Right FOV = ({:.2f}, {:.2f}) degrees".format(fov_x, fov_y))
        print("mtx =", self.camera2_mtx)
        print("dist =", self.camera2_dist)

        return ret, R, T, leftMapX, leftMapY, rightMapX, rightMapY, leftROI, rightROI, left_image_gray.shape[::-1]


if __name__ == "__main__":
    # Check if exported files are found in current directory.
    left_cam_video = Path('./left_camera.mp4')
    right_cam_video = Path('./right_camera.mp4')
    get_new_video = True
    error = False
    if left_cam_video.is_file() and right_cam_video.is_file():
        # Ask user if the want to use previously exported video files.
        use_exported = input("Would you like to use previously exported video files? (Y/n)\n")
        if use_exported == "n" or use_exported == "N":
            pass
        else:
            # Print info.
            print("User selected YES or defaulting to YES.")
            # Set toggle.
            get_new_video = False

    # Create argparser object and setup arguments.
    parser = argparse.ArgumentParser(description="Calibration utility")
    parser.add_argument("--length", "-l", type=int, default=9, help="Length of checkerboard (number of corners)")
    parser.add_argument("--width", "-w", type=int, default=6, help="Width of checkerboard (number of corners)")
    parser.add_argument("--size", "-s", type=float, default=1.0, help="Size of square")
    parser.add_argument("--output", "-o", nargs=1, default="output_params.json", help="Save the distortion constants to json file")
    parser.add_argument("--output-images", nargs=1, help="Save processed images to directory")
    parser.add_argument("--camera-A-params", "-a", required=True, help="Input file with two matrices containing the camera calibration for camera 1.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-B-params", "-b", required=True, help="Input file with two matrices containing the camera calibration for camera 2.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-A-index", "-i", required=get_new_video, help="The index of the first camera.")
    parser.add_argument("--camera-B-index", "-j", required=get_new_video, help="The index of the second camera.")
    args = parser.parse_args()

    if get_new_video:
        # Print user info, wait for input.
        input("The program will open two video windows to record chessboard calibration video. Cameras must be static\nand non moving. During video move SLOWLY and cover all areas of the image with the chessboard.\n TO STOP VIDEO PRESS 'Q'\nPress enter to continue...")
        # Grab chessboard video from both cameras.
        cap1 = cv2.VideoCapture(int(args.camera_A_index[0]))
        cap2 = cv2.VideoCapture(int(args.camera_B_index[0]))
        # Grab initial frame.
        ret1, camera1_img = cap1.read()
        ret2, camera2_img = cap2.read()

        # Check if both cameras were opened.
        if ret1 and ret2:
            # Setup video writer.
            cam1_writer = cv2.VideoWriter('left_camera.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), int(cap1.get(cv2.CAP_PROP_FPS)), (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cam2_writer = cv2.VideoWriter('right_camera.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), int(cap1.get(cv2.CAP_PROP_FPS)), (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # Continuously record video until the user presses 'q'.
            while True:
                # Grab camera images.
                ret, camera1_img = cap1.read()
                ret, camera2_img = cap2.read()

                # Check for user input.
                if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
                    cap1.release()
                    cap2.release()
                    cv2.destroyAllWindows()
                    break

                # Show images to user.
                cv2.imshow("camera1", camera1_img)
                cv2.imshow("camera2", camera2_img)
                # Write images to video files.
                cam1_writer.write(camera1_img)
                cam2_writer.write(camera2_img)

            cam1_writer.release()
            cam2_writer.release()
        else:
            # Print info and set toggle.
            print("UNABLE TO OPEN BOTH CAMERAS!")
            error = True

    if not error:
        # Split images from the video file.
        divisor = input("Enter a divisor for frame splitting. (EX) A number of 5 will store every 5th image in the video for calibration.)\n")
        # Open new video files.
        # Grab chessboard video from both cameras.
        cap1 = cv2.VideoCapture("left_camera.mp4")
        cap2 = cv2.VideoCapture("right_camera.mp4")
        # Grab initial frame.
        ret, camera1_img = cap1.read()
        ret, camera2_img = cap2.read()
        # Loop through videos until their end.
        stored_frames = 0
        total_frames = 0
        camera1_frames = []
        camera2_frames = []
        while ret:
            # Grab new frame.
            ret, camera1_img = cap1.read()
            ret, camera2_img = cap2.read()

            # Check if we should store this frame.
            if((total_frames % int(divisor)) == 0):
                # Append camera frame to list.
                camera1_frames.append(camera1_img)
                camera2_frames.append(camera2_img)
                # Increment counter.
                stored_frames += 1
            # Increment counter.
            total_frames += 1


        # Initialize are camera calibration class with the user given arguments.
        calibrate = CameraCalibration(args.width, args.length, args.size, args.camera_A_params, args.camera_B_params)
        # Check if the user want to ouput detection images to a folder.
        output_dir = args.output_images[0] if args.output_images else None
        # Call the function to detect the chessboard corners and calculate the camera coefficients.
        ret, R, T, leftMapX, leftMapY, rightMapX, rightMapY, leftROI, rightROI, image_size = calibrate.calibrateCamera(left_images=camera1_frames, right_images=camera2_frames, output_dir=output_dir)

        # Save stereo params to file.
        np.savez_compressed("./stereo_params", imageSize=image_size, leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI, rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)
        # Writing JSON data
        with open("./stereo_params.json", 'w') as f:
            json.dump({"image_size": image_size, "leftMapX": leftMapX, "leftMapY": leftMapY, "leftROI": leftROI, "rightMapX": rightMapX, "rightMapY": rightMapY, "rightROI": rightROI}, f)

