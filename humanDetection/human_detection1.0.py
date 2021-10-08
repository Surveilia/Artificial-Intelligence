#------------------------------
#Program: human_detection1.0.py
#Author:  Ben Kennedy
#Date:	  2021-10-08
#------------------------------

#imports required
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils
import numpy as np
import argparse

#This function receives a frame from the camera and then checks the frame for people.
#It does this using a histrogram of oriented gradient (HOG) descriptor algorithm to process
#the image and then uses a pre-trained model (support vector machine or SVM) to check for people.
def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride = (4,4), padding = (8,8), scale = 1.03)
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

    # Returns the frame with any detected persons bound by a rectangle
    # The frame is also returned including the total person count and SVM's confidence value
    return frame


# This function is required for capturing frames from the PiCamera
def detectByCamera():

    # Initializes the camera
    camera = PiCamera()
    camera.resolution = (320, 240)
    rawCapture = PiRGBArray(camera, size=(320, 240))

    # Tells the camera to capture video continously while storing the current frame
    # From the array into the image variable. It does this by grabbing the raw NumPy array
    # Representing the image, and then initalizing the timestamp
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

	# Check the current frame for persons
        detect(image)

	# Stream live video feed on a seperate window
        cv2.imshow("Camera Feed", image)
        key = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # Break from the loop (ending the program) when "q" is pressed
        if key == ord("q"):
            break


# Main program
if __name__ == "__main__":
    # Calls the pre-trained model for human detection 
    # Feed our SVM with it
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Start video feed
    detectByCamera()

