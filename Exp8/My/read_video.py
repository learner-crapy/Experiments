import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('slow_traffic_small.mp4')
# cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")


def TFDM(frame1, frame2, frame3, T):
    img2_1 = abs(frame2 - frame1)
    img3_2 = abs(frame3 - frame2)
    # computing the intersection of img2_1 and img3_2
    img_new = cv2.bitwise_and(img2_1, img3_2)
    img_new = cv2.threshold(img_new, T, 255, cv2.THRESH_BINARY)[1]
    return img_new

def Two(frame1, frame2):
    return abs(frame2-frame1)


frame1 = None
frame2 = None
frame3 = None
i = 0
# Read until video is completed
while (cap.isOpened()):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # gain the gray image
        if i > 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame1 = frame2
            frame2 = frame3
            frame3 = frame
            cv2.imshow('Frame', TFDM(frame1, frame2, frame3, 200))
        # Display the resulting frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i == 1:
                frame1 = frame
            if i == 2:
                frame2 = frame
            if i == 3:
                frame3 = frame
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
