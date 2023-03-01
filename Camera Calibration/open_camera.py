import cv2 as cv

# Open the camera images in current directory with glob.
cap = cv.VideoCapture(2)
    
# Use OpenCV to undistort an image from the camera. 
while True:
    # Read images with OpenCV.
    #img = cv.imread(fname)
    ret, img = cap.read()
    
    
    if cv.waitKey(1) & 0xFF == ord('q') or ret==False :
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cv.imshow("Camera", img)
