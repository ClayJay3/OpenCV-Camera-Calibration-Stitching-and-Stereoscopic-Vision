import cv2
import os
import numpy as np
import argparse

def __init__(self):
	pass

def split(image_divisor):
	x = 0
	frames = 0

	# Print debub
	print("Splitting video.mp4 into images...")

	while(True):
		# Get a new image.
		ret, frame = cap.read()
		
		if (ret == True):
			# Write the frame into the file 'output.avi'
			#out.write(frame)

			# Display the resulting frame    
			cv2.imshow('frame', frame)
			if((x % image_divisor) == 0):
				cv2.imwrite(os.path.join("Calibration Images/", "image" + str(x) + ".jpg"), frame)
				frames += 1
			x += 1

			# Press Q on keyboard to stop recording
			if cv2.waitKey(1) & 0xFF == ord('q'):
				print("Stopping the program...\nSplit a total of " + str(frames) + " images")
				break

		# Break the loop
		else:
			print("\nSplit and stored a total of " + str(frames) + " of the " + str(x) + " images.")
			break

if __name__ == '__main__':
	# Create arguments.
	parser = argparse.ArgumentParser(description='Video Split Untility')
	parser.add_argument('--divisor', '-d', type=int, default=1, help='Only store the image if its index is evenly divisible by this number.')
	parser.add_argument('input_file', type=str, help='Input video file.')

	args = parser.parse_args()
	
	# Create a VideoCapture object
	cap = cv2.VideoCapture(str(args.input_file))

	# Check if camera opened successfully
	if (cap.isOpened() == False): 
	  print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	
	split(args.divisor)

# When everything done, release the video capture and video write objects
cap.release()
#out.release()

# Closes all the frames
cv2.destroyAllWindows() 

