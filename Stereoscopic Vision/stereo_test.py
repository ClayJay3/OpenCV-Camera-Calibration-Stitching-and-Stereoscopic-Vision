import cv2
import argparse
import os.path
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt


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


def generate_depth_from_stereo_cameras(args):
    """
    This method serves as an example method for generating a depth image from two camera images.

    Parameters:
    -----------
        args - The args params from the user.
    Returns:
    --------
        Nothing
    """
    # Create instance variables.
    camera1_mtx = None
    camera1_dist = None
    camera2_mtx = None
    camera2_dist = None

    # Open the camera calibrations JSON file.
    with open(args.camera_A_params.name) as json_file:
        data = json.load(json_file)
        camera1_mtx = np.array(data["camera_matrix"])
        camera1_dist = np.array(data["distortion"])
    # Open the camera calibrations JSON file.
    with open(args.camera_B_params.name) as json_file:
        data = json.load(json_file)
        camera2_mtx = np.array(data["camera_matrix"])
        camera2_dist = np.array(data["distortion"])

    # Open the camera views.
    cap1 = cv2.VideoCapture(int(args.camera_A_index[0]))
    cap2 = cv2.VideoCapture(int(args.camera_B_index[0]))

    # Scale the camera matrix for each camera. This should allow the resolution to change without much issue.
    ret, img = cap1.read()
    if ret:
        h, w = img.shape[:2]
        camera1_mtx_scaled, roi = cv2.getOptimalNewCameraMatrix(camera1_mtx, camera1_dist, (w,h), 1, (w,h))
    else:
        print("Failed to open first camera.")
    ret, img = cap2.read()
    if ret:
        h, w = img.shape[:2]
        camera2_mtx_scaled, roi = cv2.getOptimalNewCameraMatrix(camera2_mtx, camera2_dist, (w,h), 1, (w,h))
    else:
        print("Failed to open second camera.")

    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely.
    window_size = 2
    # Creating an object of StereoBM algorithm
    left_matcher = cv2.StereoSGBM_create(
            minDisparity=-1,
            numDisparities=16*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256.
            blockSize=window_size,
            P1=9 * 3 * window_size,
            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely.
            P2=128 * 3 * window_size,
            disp12MaxDiff=12,
            uniquenessRatio=40,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    # Create opencv namedwindow so we can use trackbars.
    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities', 'disp', 1, 17, lambda: None)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, lambda: None)
    cv2.createTrackbar('preFilterCap', 'disp', 5, 62, lambda: None)
    cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, lambda: None)
    cv2.createTrackbar('speckleRange', 'disp', 0, 100, lambda: None)
    cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, lambda: None)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, lambda: None)
    cv2.createTrackbar('minDisparity', 'disp', 5, 25, lambda: None)
    cv2.createTrackbar('lambda', 'disp', 0, 100000, lambda: None)
    cv2.createTrackbar('sigma', 'disp', 0, 5, lambda: None)

    # Store start time for iterations/FPS counting.
    start_time = datetime.datetime.today().timestamp()
    iterations = 0

    # Main loop.
    while ret:
        # Grab updated camera images.
        ret, img1 = cap1.read()
        ret, img2 = cap2.read()

        # Undistort the images from both cameras using the provided camera matrix values.
        # NOTE: This might not be needed. Stereo remap might do this on its own given the parameters. Undistoring twice is probably my issue.
        # camera1_img = cv2.undistort(img1, camera1_mtx, camera1_dist, None, camera1_mtx_scaled)
        # camera2_img = cv2.undistort(img2, camera2_mtx, camera2_dist, None, camera2_mtx_scaled)
        camera1_img_gray = cv2.cvtColor(camera1_img, cv2.COLOR_BGR2GRAY)
        camera2_img_gray = cv2.cvtColor(camera2_img, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow("camera1", camera1_img_gray)
        cv2.imshow("camera2", camera2_img_gray)
    
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')
        lmbda = cv2.getTrackbarPos('lambda', 'disp')
        sigma = cv2.getTrackbarPos('sigma', 'disp')
        
        # Setting the updated parameters before computing disparity map
        left_matcher.setNumDisparities(numDisparities)
        left_matcher.setBlockSize(blockSize)
        left_matcher.setPreFilterCap(preFilterCap)
        left_matcher.setUniquenessRatio(uniquenessRatio)
        left_matcher.setSpeckleRange(speckleRange)
        left_matcher.setSpeckleWindowSize(speckleWindowSize)
        left_matcher.setDisp12MaxDiff(disp12MaxDiff)
        left_matcher.setMinDisparity(minDisparity)

        # Apply filter to reduce noise and make depth map more even.
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        try:
            # Compute disparity
            displ = left_matcher.compute(camera1_img_gray, camera2_img_gray)  # .astype(np.float32)/16
            dispr = right_matcher.compute(camera2_img_gray, camera1_img_gray)  # .astype(np.float32)/16
            displ = np.int16(displ)
            dispr = np.int16(dispr)
            filteredImg = wls_filter.filter(displ, camera1_img_gray, None, dispr)  # important to put "imgL" here!!!
            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)

            # Displaying the disparity map
            cv2.imshow("image", filteredImg)
        except Exception as e:
            print(e)


        # Increment FPS and print.
        time_diff = datetime.datetime.today().timestamp() - start_time
        iterations += 1
        print(iterations / time_diff)


if __name__ == "__main__":
    # Setup argparse arguments.
    parser = argparse.ArgumentParser(description="Example Point cloud generator from two images.")
    parser.add_argument("--camera-A-params", "-a", required=True, help="Input file with two matrices containing the camera calibration for camera 1.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-B-params", "-b", required=True, help="Input file with two matrices containing the camera calibration for camera 2.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-A-index", "-i", required=True, help="The index of the first camera.")
    parser.add_argument("--camera-B-index", "-j", required=True, help="The index of the second camera.")
    args = parser.parse_args()

    # Start point cloud generation.
    generate_depth_from_stereo_cameras(args)