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

def showImg(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def write_ply(fn, verts, colors):
    """
    This method generates a point cloud file from a 3D list of points and a 2D list of colors for each pixel.

    Parameters:
    -----------
        fn - The output file name.
        verts - The actual points of the point cloud.
        colors - The colors for each point.
    Returns:
    --------
        Nothing
    """
    # Create instance variables.
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    # Assemble and write point cloud file.
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    # Attach colors to end of 3rd dimension.
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def generate_point_cloud_from_stereo_cameras(args):
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

    # Creates window
    cv2.namedWindow('stereo_disparity')
    # Creates Trackbar with slider position and callback function
    cv2.createTrackbar('numDisparities', 'stereo_disparity', 48, 200, (lambda: None))
    cv2.createTrackbar('blockSize', 'stereo_disparity', 5, 200, (lambda: None))

    # Store start time for iterations/FPS counting.
    start_time = datetime.datetime.today().timestamp()
    iterations = 0

    # Main loop.
    while ret:
        # Grab updated camera images.
        ret, img1 = cap1.read()
        ret, img2 = cap2.read()

        # Undistort the images from both cameras using the provided camera matrix values.
        camera1_img = cv2.undistort(img1, camera1_mtx, camera1_dist, None, camera1_mtx_scaled)
        camera2_img = cv2.undistort(img2, camera2_mtx, camera2_dist, None, camera2_mtx_scaled)
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
        try:
            stereo = cv2.StereoBM_create(numDisparities=cv2.getTrackbarPos('numDisparities', 'stereo_disparity'), blockSize=cv2.getTrackbarPos('blockSize', 'stereo_disparity'))
            disparity = stereo.compute(camera1_img_gray, camera2_img_gray)
            plt.imshow(disparity, 'CMRmap_r')
            plt.pause(0.05)
        except Exception:
            print("numDisparities or blackSize is invalid, try adjusting one of them.")

        # # Transform the disparity map so that we can obtain depth information, to do this we need the disparity-to-depth matrix.
        # cam1_calib_mtx = camera1_mtx[:,:3] # Left color image.
        # cam2_calib_mtx = camera2_mtx[:,:3] # Right color image.
        # rev_proj_matrix = np.zeros((4,4)) # To store the output.
        # Tmat = np.array([0.54, 0., 0.])
        # # Calculate depth matrix.
        # cv2.stereoRectify(cameraMatrix1 = cam1_calib_mtx,cameraMatrix2 = cam2_calib_mtx,
        #                 distCoeffs1 = 0, distCoeffs2 = 0,
        #                 imageSize = camera1_img.shape[:2],
        #                 R = np.identity(3), T = Tmat,
        #                 R1 = None, R2 = None,
        #                 P1 =  None, P2 =  None, 
        #                 Q = rev_proj_matrix)

        # # Project disparity map to 3D point cloud.
        # points = cv2.reprojectImageTo3D(disparity, rev_proj_matrix)

        # #reflect on x axis
        # reflect_matrix = np.identity(3)
        # reflect_matrix[0] *= -1
        # points = np.matmul(points, reflect_matrix)

        # #extract colors from image
        # colors = cv2.cvtColor(camera1_img, cv2.COLOR_BGR2RGB)

        # #filter by min disparity
        # mask = disparity > disparity.min()
        # out_points = points[mask]
        # out_colors = colors[mask]

        # #filter by dimension
        # idx = np.fabs(out_points[:,0]) < 4.5
        # out_points = out_points[idx]
        # out_colors = out_colors.reshape(-1, 3)
        # out_colors = out_colors[idx]
        # # Write point cloud file.
        # # write_ply('out.ply', points, out_colors)

        # # Project points and colors onto 2D image for easy viewing.
        # reflected_pts = np.matmul(points, reflect_matrix)
        # projected_img,_ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]), cam2_calib_mtx[:3,:3], np.array([0., 0., 0., 0.]))
        # projected_img = projected_img.reshape(-1, 2)

        # blank_img = np.zeros(camera1_img.shape, 'uint8')
        # img_colors = camera2_img[mask][idx].reshape(-1,3)

        # for i, pt in enumerate(projected_img):
        #     pt_x = int(pt[0])
        #     pt_y = int(pt[1])
        #     if pt_x > 0 and pt_y > 0:
        #         # use the BGR format to match the original image type
        #         col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
        #         cv2.circle(blank_img, (pt_x, pt_y), 1, col)

        # # Show point cloud image.
        # showImg(blank_img)

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
    generate_point_cloud_from_stereo_cameras(args)