# Import necessary packages and libraries.
import cv2
import argparse
import imutils
from imutils import paths
import time
import numpy as np


# Main program function.
class ImageStitcher(object):
    '''
    This class provides an easy and robust image stitcher using opencv.
    It takes in a single image at a time or a list of images, and will attempt to stitch them together.
    The user can set multiple options to better fit their image stitching needs.
    '''
    def __init__(self, mode="PANORAMA") -> None:
        '''
        Class initialization function. Setups up member variables and objects.

        Parameters:
        -----------
            mode - A string. Either PANORAMA or SCANS. Scans is better for top-down photos when building maps.

        Returns:
        --------
            Nothing
        '''
        # Create class variables.
        self.images = []
        self.mode = cv2.STITCHER_SCANS if mode == "SCANS" else cv2.STITCHER_PANORAMA

        # Create class objects.
        self.stitcher = cv2.Stitcher.create(self.mode) if imutils.is_cv3() else cv2.Stitcher_create(self.mode)

    def add_image(self, *images) -> None:
        '''
        Takes a single image or list of images and adds them to the class images list.

        Parameters:
        -----------
            *images - An opencv mat image or 2D list. OR a list of mat images or 2D lists.
                    OR multiple images given like image1, image2, image3, image4.

        Returns:
        --------
            bool - Whether or not all the images were successfully stored.
        '''
        # Create function instance variables.
        images_stored = True

        # Check if image is not empty and contains data.
        if images is None or len(images) <= 0:
            images_stored = False
        # Check if the function was passed just one image.
        elif len(np.array(images).shape) == 2:
            # Store image.
            self.images.append(images)
        elif len(np.array(images).shape) > 2:
            # Loop through each image.
            for image in images:
                # Check image if image is not empty.
                if image is not None:
                    # Store image.
                    self.images.append(image)
                else:
                    # An image failed to be stored.
                    images_stored = False
        else:
            # An image failed to be stored.
            images_stored = False

        # Return image store status.
        return images_stored

    def stitch_images(self):
        """
        This function takes the images stored in the classes images list, and attempts to
        stitch them together using the stitch mode choosen by the user when the class was 
        initialized.

        Parameters:
        -----------
            None

        Returns:
        --------
            string - The return status of the stitcher according to the Status enumerator in the Stitcher class.
                    [ OK = 0 | ERR_NEED_MORE_IMGS = 1 | ERR_HOMOGRAPHY_EST_FAIL = 2 | ERR_CAMERA_PARAMS_ADJUST_FAIL = 3 ]
            output - The output image.
        """
        # Create method instance variables.
        result_status = "OK"

        # Attempt to stitch images.
        (status, stitched_image) = self.stitcher.stitch(self.images)

        # Parse status.
        if status == 1:
            result_status = "ERR_NEED_MORE_IMGS"
        elif status == 2:
            result_status = "ERR_HOMOGRAPHY_EST_FAIL"
        elif status == 3:
            result_status = "ERR_CAMERA_PARAMS_ADJUST_FAIL"

        # Return results
        return result_status, stitched_image


def main(args):
    """
    Main method. Sets up stitcher class and executes stitch method.

    Parameters:
    -----------
        args - The list of arguments from the argparser.

    Returns:
    --------
        Nothing
    """
    # Construct image stitch class.
    stitcher = ImageStitcher(mode=args["mode"])

    # Grab the paths to the input images and initialize our images list
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images("images")))

    # Loop through images paths.
    for path in imagePaths:
        # Open image.
        image = cv2.imread(path)
        # Add image to stitcher class.
        stitcher.add_image(image)

    # Stitch images.
    status, stitch = stitcher.stitch_images()
    # Print status and show image.
    print(f"Image Stitch Return Status: {status}")

    # Check if final image is empty.
    if stitch is not None and len(stitch) > 0:
        cv2.imshow("STITCHED", stitch)
        cv2.imwrite(f"outputs/{time.strftime('%Y%m%d-%H%M%S')}.jpg", np.array(stitch))
        cv2.waitKey(0)
    else:
        if status == "OK":
            # Print error message.
            print("ERROR: Stitch image is empty! Did you put any images in the images folder?")


# Run main function.
if __name__ == "__main__":
    # Create the argument parser and add arguments.
    argpar = argparse.ArgumentParser()
    argpar.add_argument("-m", "--mode", type=str, required=True, help="Stitching method to use. (PANORAMA | SCANS)")
    args = vars(argpar.parse_args())

    # Call main method and pass args.
    try:
        main(args)
    except Exception as error:
        print(error)
        print("The program errored out. Make sure the images are in order from left-right top-bottom and are in the same orientation.")
