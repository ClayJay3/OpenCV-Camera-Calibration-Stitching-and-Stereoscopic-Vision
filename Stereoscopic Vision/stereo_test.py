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


def generate_point_cloud_from_stereo_cameras(args, calibration):
    """
    This method serves as an example method for generating a point cloud from two camera images.

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

    imageSize = tuple(calibration["imageSize"])
    leftMapX = calibration["leftMapX"]
    leftMapY = calibration["leftMapY"]
    leftROI = tuple(calibration["leftROI"])
    rightMapX = calibration["rightMapX"]
    rightMapY = calibration["rightMapY"]
    rightROI = tuple(calibration["rightROI"])

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

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()
    # Create opencv namedwindow so we can use trackbars.
    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities','disp',1,17,lambda: None)
    cv2.createTrackbar('blockSize','disp',5,50,lambda: None)
    cv2.createTrackbar('preFilterType','disp',1,1,lambda: None)
    cv2.createTrackbar('preFilterSize','disp',2,25,lambda: None)
    cv2.createTrackbar('preFilterCap','disp',5,62,lambda: None)
    cv2.createTrackbar('textureThreshold','disp',10,100,lambda: None)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,lambda: None)
    cv2.createTrackbar('speckleRange','disp',0,100,lambda: None)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,lambda: None)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,lambda: None)
    cv2.createTrackbar('minDisparity','disp',5,25,lambda: None)

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

        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow("camera1", camera1_img)
        cv2.imshow("camera2", camera2_img)

        # Compute the pixel disparity map. This is the distance difference between corresponding pixels in the image.
        # Applying stereo image rectification on the left image
        left_nice= cv2.remap(camera1_img_gray,
                leftMapX,
                leftMapY,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        right_nice= cv2.remap(camera2_img_gray,
                rightMapX,
                rightMapY,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
    
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
    
        try:
        #   Calculating disparity using the StereoBM algorithm
            disparity = stereo.compute(left_nice, right_nice)
        except Exception:
            pass
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
    
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities
    
        # Displaying the disparity map
        cv2.imshow("image",disparity)

        # Increment FPS and print.
        time_diff = datetime.datetime.today().timestamp() - start_time
        iterations += 1
        print(iterations / time_diff)


if __name__ == "__main__":
    # Setup argparse arguments.
    parser = argparse.ArgumentParser(description="Example Point cloud generator from two images.")
    parser.add_argument("--camera-A-params", "-a", required=True, help="Input file with two matrices containing the camera calibration for camera 1.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-B-params", "-b", required=True, help="Input file with two matrices containing the camera calibration for camera 2.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--stereo-camera-params", "-c", required=True, help="Input file with matrices containing the cameras' stereo calibration.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-A-index", "-i", required=True, help="The index of the first camera.")
    parser.add_argument("--camera-B-index", "-j", required=True, help="The index of the second camera.")
    args = parser.parse_args()

    # Load stereo calibration for cameras.
    calibration = np.load(args.stereo_camera_params.name, allow_pickle=False)

    # Start point cloud generation.
    generate_point_cloud_from_stereo_cameras(args, calibration)