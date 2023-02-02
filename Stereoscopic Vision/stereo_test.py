import cv2
import argparse
import os.path


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


def generate_point_cloud_from_stereo_cameras():
    """
    This method serves as an example method for generating a point cloud from two camera images.

    Parameters:
    -----------

    Returns:
    --------

    """


if __name__ == "__main__":
    # Setup argparse arguments.
    parser = argparse.ArgumentParser(description="Example Point cloud generator from two images.")
    parser.add_argument("--camera-A-params", "-a", required=True, help="Input file with two matrices containing the camera calibration for camera 1.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-B-params", "-b", required=True, help="Input file with two matrices containing the camera calibration for camera 2.", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--camera-A-index", "-i", required=True, help="The index of the first camera.")
    parser.add_argument("--camera-B-index", "-j", required=True, help="The index of the second camera.")