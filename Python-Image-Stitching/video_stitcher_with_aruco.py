import cv2
import numpy
import os.path
import argparse
import json
import numpy as np
import datetime
import pyzed.sl as sl
from obstacle_detector import YOLOObstacleDetector


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



class VideoStitcher(object):
    """
    Calculate homography of both cameras and stitches their video together.
    """

    def __init__(self):
        """
        Declares and initializes member variables and objects.
        """
        # Create member variables.
        self.homo_matrix = np.asarray([[ 1.20254541e+00,  1.46581672e-01,  3.97339938e+02], [ 1.32351777e-01,  1.10367117e+00, -9.12597223e+00], [ 5.45140490e-04,  1.93311732e-04,  1.00000000e+00]])
        self.B = None

    def stitch(self, images, ratio=0.75, reproj_thresh=4.0):
        """
        This function takes in an number of different camera images and,
        if a homography matrix hasn't been generated yet, it will attempt to
        calculate one. Upon successfully created an homography matrix, the camera
        images will be warped/bent to become stitched together according to the homography.

        Parameters:
        -----------
            images - A list of image images to stitch together.
            ratio - The ratio to multiply the difference between the matches by.
            reproj_thresh - The pixel thresh to give the OpenCV matchkeypoints function.

        Returns:
        --------
            result - The stitched image.
        """
        # Pull out images from list.
        image_a, image_b = images[0], images[1]

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it.
        if self.homo_matrix is None:
            # Detect keypoints and extract.
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = self.calculate_homography(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)
            print(matched_keypoints)

            # If the match is None, then there aren't enough matched keypoints to create a panorama.
            if matched_keypoints is None:
                return None

            # Save the homography matrix.
            self.homo_matrix = matched_keypoints[1]

        # Apply a perspective transform to stitch the images together using the saved homography matrix.
        output_shape = (image_a.shape[1] + image_b.shape[1], image_a.shape[0] + image_b.shape[1])
        tranformed_image = cv2.warpPerspective(image_a, self.homo_matrix, output_shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Get a mask of the tranformed image.
        img_gray = cv2.cvtColor(tranformed_image, cv2.COLOR_BGR2GRAY)
        tranformed_mask = numpy.zeros(shape=img_gray.shape, dtype=np.uint8)
        tranformed_mask = np.expand_dims(np.where(img_gray == 0, 255, tranformed_mask), axis=2)
        # Erode the mask to prevent black stitch lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tranformed_mask = cv2.morphologyEx(tranformed_mask, cv2.MORPH_DILATE, kernel)

        # Resize image_b to match the transformed image_a res.
        image_b = cv2.copyMakeBorder(image_b, top=0, bottom=(output_shape[1] - image_b.shape[0]), left=0, right=(output_shape[0] - image_b.shape[1]), borderType=cv2.BORDER_CONSTANT)

        # Perform bitwise AND on the untransformed image to prepare for overlaying transformed image.
        image_b = cv2.bitwise_and(image_b, image_b, mask=tranformed_mask)
        # Perform bitwise OR to stitch the two images together.
        result = cv2.bitwise_or(image_b, tranformed_image)

        # Return the stitched image
        return result

    def detect_and_extract(self, image):
        """
        This function takes in an image and will run some OpenCV alogrithms to
        extract the features from the image. These features are hard edges,
        big points, or any other notable geometry.

        Parameters:
        -----------
            image - The image to extract features on.

        Returns:
        --------
            (keypoints, features) - The points and lines of the image.
        """
        # Detect and extract features from the image. (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays.
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features.
        return (keypoints, features)

    def calculate_homography(self, keypoints_a, keypoints_b, features_a, features_b, ratio=0.75, reproj_thresh=4.0):
        """
        This method takes in the keypoints and features from two cameras,
        and uses the bruteforce method of RANSAC to allign the two images.

        Parameters:
        -----------
            keypoints_a - Extracted keypoints from the first image.
            keypoints_b - Extracted keypoints from the second image.
            features_a - Extracted features from the first image.
            features_b - Extracted features from the second image.
            ratio - The ratio to multiply the difference between the matches by.
            reproj_thresh - The pixel thresh to give the OpenCV matchkeypoints function.

        Returns:
        --------
            matches - An array of matched features.
            homography matrix - The calculated homography matrix
            status - The return status of the findHomography OpenCV function.

            (Returns None if homography failed.)
        """
        # Compute the raw matches and initialize the list of actual matches.
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawmatches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for rawmatch in rawmatches:
            # Ensure the distance is within a certain ratio of each other. (i.e. Lowe's ratio test)
            if len(rawmatch) == 2 and rawmatch[0].distance < rawmatch[1].distance * ratio:
                matches.append((rawmatch[0].trainIdx, rawmatch[0].queryIdx))

        # Computing a homography requires at least 4 matches.
        if len(matches) > 4:
            # Construct the two sets of points.
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points.
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point.
            return (matches, homography_matrix, status)

        # No homography could be computed.
        return None

    def project_onto_cylinder(self, img, K_matrix):
        """
        Projects the given image onto a cylinder, this makes stitching of cameras less planar.

        Parameters:
        -----------
            img - The user given image to project.
            K_matrix - The K matrix from camera calibration.

        Returns:
        --------
            transformed_image - The resulting image.
        """
        img_h, img_w = img.shape[:2]

        # Check if we need to calculate B matrix for projection.
        if self.B is None:
            # Pixel coordinates.
            y_i, x_i = np.indices((img_h, img_w))
            X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(img_h * img_w, 3)  # To Homography with the given K matrix.
            K_inv = np.linalg.inv(K_matrix)
            X = K_inv.dot(X.T).T    # Normalized coords.
            # Calculate cylindrical coords. (sin\theta, h, cos\theta)
            A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(img_h * img_w, 3)
            B = K_matrix.dot(A.T).T    # Project back to image-pixels plane.
            # Back from homog coords.
            B = B[:, :-1] / B[:, [-1]]
            # Make sure warp coords only within image bounds.
            B[(B[:, 0] < 0) | (B[:, 0] >= img_w) | (B[:, 1] < 0) | (B[:, 1] >= img_h)] = -1
            self.B = B.reshape(img_h, img_w, -1)
        
        # Tranform image onto cylinder coordinate system.
        img = cv2.remap(img, self.B[:, :, 0].astype(np.float32), self.B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)

        # Warp the image according to cylindrical coords.
        return img


if __name__ == "__main__":
    # Create argparser object and setup arguments.
    parser = argparse.ArgumentParser(description="Calibration utility")
    parser.add_argument("--camera-A-params", "-a", required=True, help="Input file with two matrices containing the camera calibration for camera 1.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-B-params", "-b", required=True, help="Input file with two matrices containing the camera calibration for camera 2.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-C-params", "-c", required=True, help="Input file with two matrices containing the camera calibration for camera 3.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--fisheye', '-f', type=bool, default=False, help='If the camera calibrations are for fisheye lenses.')
    parser.add_argument('--lefteye', '-l', type=bool, default=False, help='Whether or not to use the left or right camera.')
    parser.add_argument("--camera-A-source", "-i", required=True, help="The index of the first camera.")
    parser.add_argument("--camera-B-source", "-j", required=True, help="The index of the second camera.")
    args = parser.parse_args()

    # Check if the calibrations or formatted for fisheye cameras.
    if not args.fisheye:
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
        # Open the camera calibrations JSON file.
        if args.camera_C_params.name is not None:
            with open(args.camera_C_params.name) as json_file:
                data = json.load(json_file)
                camera3_mtx = np.array(data["camera_matrix"])
                camera3_dist = np.array(data["distortion"])
    else:
        # Open the camera calibrations JSON file.
        with open(args.camera_A_params.name) as json_file:
            data = json.load(json_file)
            camera1_K = np.array(data["K_matrix"])
            camera1_D = np.array(data["D_matrix"])
            camera1_DIM = np.array(data["DIM"])
        # Open the camera calibrations JSON file.
        with open(args.camera_B_params.name) as json_file:
            data = json.load(json_file)
            camera2_K = np.array(data["K_matrix"])
            camera2_D = np.array(data["D_matrix"])
            camera2_DIM = np.array(data["DIM"])
        # Open the camera calibrations JSON file.
        if args.camera_C_params.name is not None:
            with open(args.camera_C_params.name) as json_file:
                data = json.load(json_file)
                camera3_mtx = np.array(data["camera_matrix"])
                camera3_dist = np.array(data["distortion"])

    # Open the camera views.
    cap1 = cv2.VideoCapture(int(args.camera_A_source[0]))
    cap2 = cv2.VideoCapture(int(args.camera_B_source[0]))

    # Scale the camera matrix for each camera. This should allow the resolution to change without much issue.
    ret, img = cap1.read()
    if ret:
        # Check if this is a fisheye camera.
        if not args.fisheye:
            # Normal Camera.
            h, w = img.shape[:2]
            camera1_mtx_scaled, roi = cv2.getOptimalNewCameraMatrix(camera1_mtx, camera1_dist, (w, h), 1, (w, h))
        else:
            # Fisheye camera.

            # Zoom out some, don't crop off as much of image.
            # camera1_K_scaled = camera1_K.copy()
            # camera1_K_scaled[0,0]=camera1_K[0,0]/2
            # camera1_K_scaled[1,1]=camera1_K[1,1]/2
            # Calculate camera distortion maps.
            camera1_map1, camera1_map2 = cv2.fisheye.initUndistortRectifyMap(camera1_K, camera1_D, np.eye(3), camera1_K, camera1_DIM, cv2.CV_16SC2)
    else:
        print("Failed to open first camera.")
    ret, img = cap2.read()
    if ret:
        # Check if this is a fisheye camera.
        if not args.fisheye:
            # Normal camera.
            h, w = img.shape[:2]
            camera2_mtx_scaled, roi = cv2.getOptimalNewCameraMatrix(camera2_mtx, camera2_dist, (w, h), 1, (w, h))
        else:
            # Fisheye camera.

            # Zoom out some, don't crop off as much of image.
            # camera2_K_scaled = camera2_K.copy()
            # camera2_K_scaled[0,0]=camera2_K[0,0]/2
            # camera2_K_scaled[1,1]=camera2_K[1,1]/2
            # Calculate camera distortion maps.
            camera2_map1, camera2_map2 = cv2.fisheye.initUndistortRectifyMap(camera2_K, camera2_D, np.eye(3), camera2_K, camera2_DIM, cv2.CV_16SC2)
    else:
        print("Failed to open second camera.")

    # Only attempt to open zed camera if we were given a parameter.
    if args.camera_C_params.name is not None:
        # Attempt to open a zed camera.
        # Create a ZED camera object
        zed = sl.Camera()
        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Unable to open ZED camera.")
            exit(-1)
        else:
            # Get ZED data.
            image_size = zed.get_camera_information().camera_resolution
            # Create custom mat for zed image.
            image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
                
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

                # Normal camera.
                h, w = img.shape[:2]
                camera3_mtx_scaled, roi = cv2.getOptimalNewCameraMatrix(camera3_mtx, camera3_dist, (w, h), 1, (w, h))


    """ 
    This section of code determines how much of the image needs to be cutoff to 
    remove the black borders from undistortion.
    """
    # Create video stitcher.
    stitcher = VideoStitcher()

    # Grab initial camera images.
    ret, img1 = cap1.read()
    ret, img2 = cap2.read()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Check if we should grab images from the left or right camera.
        if (args.lefteye):
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Retrieve the left image
        else:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image_zed, sl.VIEW.RIGHT) # Retrieve the left image

        # Get camera frame from zed.
        img3 = image_zed.get_data()
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGRA2BGR)

    # Check if we are using fisheye cameras.
    camera1_img, camera2_img, camera3_img = None, None, None
    if not args.fisheye:
        # Undistort the images from both cameras using the provided camera matrix values.
        camera1_img = cv2.undistort(img1, camera1_mtx, camera1_dist, None, camera1_mtx_scaled)
        camera2_img = cv2.undistort(img2, camera2_mtx, camera2_dist, None, camera2_mtx_scaled)
        camera3_img = cv2.undistort(img3, camera3_mtx, camera3_dist, None, camera3_mtx_scaled)
    else:
        # Undistort the images from both cameras using the provided camera matrix values for fisheye.
        camera1_img = cv2.remap(img1, camera1_map1, camera1_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        camera2_img = cv2.remap(img2, camera2_map1, camera2_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        camera3_img = cv2.undistort(img3, camera3_mtx, camera3_dist, None, camera3_mtx_scaled)
        camera1_img = stitcher.project_onto_cylinder(camera1_img, camera1_K)
        camera2_img = stitcher.project_onto_cylinder(camera2_img, camera2_K)
        camera3_img = stitcher.project_onto_cylinder(camera3_img, camera3_mtx)

    # Loop through both images.
    image_crops = []
    for img in (camera1_img, camera2_img, camera3_img):
        # Get image dimensions.
        h, w = img.shape[0], img.shape[1]
        # Get middle row and column from image.
        middle_row = img[h // 2]
        middle_column = img[:, w // 2]
        # Loop through row and find black borders.
        x_min, x_max = 0, w
        for i in range(w // 2):
            # Check min row.
            if (middle_row[(w // 2) - i] == [0, 0, 0]).all() and x_min == 0:
                x_min = (w // 2) - i
            # Check max row.
            if (middle_row[(w // 2) + i] == [0, 0, 0]).all() and x_max == w:
                x_max = (w // 2) + i
        # Loop through column and find black borders.
        y_min, y_max = 0, h
        for i in range(h // 2):
            # Check min row.
            if (middle_column[(h // 2) - i] == [0, 0, 0]).all() and y_min == 0:
                y_min = (h // 2) - i
            # Check max row.
            if (middle_column[(h // 2) + i] == [0, 0, 0]).all() and y_max == h:
                y_max = (h // 2) + i

        # Append x and y limits to list.
        image_crops.append([x_min, x_max, y_min, y_max])

    print("[INFO] Cropping images to (removes black borders): ", image_crops)

    # Store start time for iterations/FPS counting.
    start_time = datetime.datetime.today().timestamp()
    iterations = 0

    # Setup aruco library.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.markerBorderBits = 1
    parameters.errorCorrectionRate = 1
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    r_vec = None
    t_vec = None

    yolo_detector = YOLOObstacleDetector(os.path.dirname(__file__) + "/best.pt", 640, 0.4)

    # Main loop.
    while True:
        # Create instance variables.
        cropped_images = []

        # Grab updated camera images.
        ret, img1 = cap1.read()
        ret, img2 = cap2.read()
        img3 = None
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Check if we should grab images from the left or right camera.
            if (args.lefteye):
                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Retrieve the left image
            else:
                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image_zed, sl.VIEW.RIGHT) # Retrieve the left image

            # Get camera frame from zed.
            img3 = image_zed.get_data()
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGRA2BGR)

        # Check if we are using fisheye cameras.
        camera1_img, camera2_img, camera3_img = None, None, None
        if not args.fisheye:
            # Undistort the images from both cameras using the provided camera matrix values.
            camera1_img = cv2.undistort(img1, camera1_mtx, camera1_dist, None, camera1_mtx_scaled)
            camera2_img = cv2.undistort(img2, camera2_mtx, camera2_dist, None, camera2_mtx_scaled)
            camera3_img = cv2.undistort(img3, camera3_mtx, camera3_dist, None, camera3_mtx_scaled)

            # Check if camera image is not None. If not, then put zed image inbetween fisheyes.
            if camera3_img is not None:
                # Crop images.
                for crop, image in zip(image_crops, [camera1_img, camera2_img, camera3_img]):
                    cropped_images.append(image[crop[2]:crop[3], crop[0]:crop[1]].copy())
            else:
                # Crop images.
                for crop, image in zip(image_crops, [camera1_img, camera2_img]):
                    cropped_images.append(image[crop[2]:crop[3], crop[0]:crop[1]].copy())
        else:
            # Undistort the images from both cameras using the provided camera matrix values for fisheye.
            camera1_img = cv2.remap(img1, camera1_map1, camera1_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            camera2_img = cv2.remap(img2, camera2_map1, camera2_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            camera3_img = cv2.undistort(img3, camera3_mtx, camera3_dist, None, camera3_mtx_scaled)
            camera1_img = stitcher.project_onto_cylinder(camera1_img, camera1_K)
            camera2_img = stitcher.project_onto_cylinder(camera2_img, camera2_K)

            # Check if camera image is not None. If not, then put zed image inbetween fisheyes.
            if camera3_img is not None:
                # Crop images.
                for crop, image in zip(image_crops, [camera1_img, camera2_img, camera3_img]):
                    cropped_images.append(image[crop[2]:crop[3], crop[0]:crop[1]].copy())
            else:
                # Crop images.
                for crop, image in zip(image_crops, [camera1_img, camera2_img]):
                    cropped_images.append(image[crop[2]:crop[3], crop[0]:crop[1]].copy())

        # Don't need to crop images.
        # cropped_images.append(camera1_img)
        # cropped_images.append(camera2_img)

        stitched_image = stitcher.stitch([cropped_images[1], cropped_images[0]], ratio=0.75, reproj_thresh=2.5)
        # Run yolo.
        yolo_detector.detect_obstacles(stitched_image)
        object_summary = yolo_detector.track_obstacle(stitched_image)
        cv2.imshow("Result1", stitched_image)

        # Test Aruco Tag detection.
        corners, ids, rejectedImgPoints = detector.detectMarkers(camera3_img)

        # Add Tags to Tag Class Object
        if ids is not None:
            reg_img = cv2.aruco.drawDetectedMarkers(camera3_img, corners)

            # Convert corners into 1D array of tuples.
            imagePoints = np.asarray(corners[0][0], dtype=np.float32)
            objectPoints = np.asarray([[0, 0, 0],[10, 0, 0],[10, 10, 0],[0, 10, 0]], dtype=np.float32)
            cameraMatrix = np.asarray([[239.4694816151291, 0.0, 301.67024898194353], [0.0, 239.5116789433278, 243.12508520685762], [0.0, 0.0, 1.0]], dtype=np.float32)
            disMatrix = np.asarray([[0, 0, 0, 0, 0]], dtype=np.float32)

            # if r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                objectPoints, imagePoints, cameraMatrix, disMatrix)
            r_vec = rotation_vector
            t_vec = translation_vector
            cv2.drawFrameAxes(camera3_img, cameraMatrix, disMatrix, r_vec, t_vec, 15)

        cv2.imshow("Aruco", camera3_img)

        # Increment FPS and print.
        time_diff = datetime.datetime.today().timestamp() - start_time
        iterations += 1
        # print("FPS: ", int(iterations / time_diff))

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            break