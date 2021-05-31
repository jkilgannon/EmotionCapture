"""
 
CISC 642
Final Project
Capture video feed
Jon Kilgannon
 
"""

import sys

# User will send in a request to find all faces via the command line.
#   If no command line argument is sent, then the user wants to find
#   only one face.
if len(sys.argv) == 1:
    find_only_one_face = True
elif sys.argv[1].strip().upper() == "-ALL":
    find_only_one_face = False 
else:
    print("To detect the largest face, run this program with no command line arguments.\nTo detect all faces, use the '-all' command line argument.\nYour command line argument was not recognized.")
    sys.exit()

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
   

# Prebuilt Haar detector
haar_loc = "haarcascade_frontalface_default.xml"
facedetector = cv2.CascadeClassifier(haar_loc)

## One emotion-detecting neural network
#emotion_net_loc = "emotions.h5"
#emotion_net = keras.models.load_model(emotion_net_loc)

# Load all of my emotion nets
emotion_nets = []
for file in os.listdir("."):
    if file.endswith(".h5"):
        emotion_net = keras.models.load_model(file)
        emotion_nets.append(emotion_net)
        print("Using Emotion Net '" + file + "'")


# Info for the text we put on screen
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color_one_face = (0, 255, 0)
color_all_faces = (255, 255, 255)
thickness = 2

# The text versions of the predicted emotions; used for on-screen display
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprised"]

def get_predicted_emotion(possible_face):
    # Takes in an image that the Haar detector thinks is a face, and predicts an emotion for it
    #   This works by taking a vote among the Emotion Nets
    
    all_predictions = []
    
    for e_n in emotion_nets:
        # Let each neural net predict
        resize = cv2.resize(possible_face, (48,48), interpolation = cv2.INTER_AREA)
        npary = np.array(resize)
        npary = npary.reshape((1,48,48,1))      # Make it into a one-item batch for the network to digest
        pred = e_n.predict(npary)
        emotion_prediction = np.argmax(pred, axis=-1)
        emotion_prediction= emotion_prediction[0]          # It's returned in a one-element array
        all_predictions.append(emotion_prediction)

    # Vote!
    prediction_counts = [0,0,0,0,0,0,0]
    for prediction in all_predictions:
        prediction_counts[prediction] = prediction_counts[prediction] + 1 
        
    if prediction_counts.count(1) == len(prediction_counts):
        # Everyone voted differently.  There's no consensus
        max_votes = -1
    else:
        max_votes = np.argmax(prediction_counts)

    if max_votes == -1:
        emotion_title = "No consensus"
    else:
        emotion_title = emotions[emotion_prediction]

    return emotion_title


## Capture video, get faces, determine emotions in the faces

# Capture video from camera  
vidcap = cv2.VideoCapture(0)

# We will capture frames until the user hits X
exit_wanted = False
while(not exit_wanted):
    _, image = vidcap.read()
    
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    min_allowable_box = (100,100)
    faceboxes = facedetector.detectMultiScale(grayimg, scaleFactor=1.05,
                                              minNeighbors=5, minSize=min_allowable_box,
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    
    if find_only_one_face:
        # Threshold to find the human: Find the largest face
        biggest_area = 0
        best_x = 0
        best_y = 0
        best_width = 1
        best_height = 1
        
        for (x, y, width, height) in faceboxes:
            if (width * height) > biggest_area:
                 biggest_area = width * height
                 best_x = x
                 best_y = y
                 best_width = width
                 best_height = height
    
        # Don't try to show anything if there's nothing to show
        if biggest_area >= 100*100:
            # Well, we found SOMETHING.  Work with it.
            
            # Get the grayscale of the largest "face" as its own image
            cropped_face = grayimg[best_y:best_y + best_height, best_x:best_x + best_width]
            
            # Draw a box around just that face on the color image.
            cv2.rectangle(image, (best_x, best_y), (best_x + best_width, best_y + best_height), (0, 255, 0), 2)
            
            emotion_prediction = get_predicted_emotion(cropped_face)
            
            # Put text on the color image
            text_loc = (best_x + 5, best_y - 15)  # location on image
            image = cv2.putText(image, emotion_prediction, text_loc, font, 
                               fontScale, color_one_face, thickness, cv2.LINE_AA)
        else:
            # We didn't find anything
            text_loc = (30, 30)
            image = cv2.putText(image, "No face captured", text_loc, font, 
                               fontScale, color_one_face, thickness, cv2.LINE_AA)
    else:
         # Display all the faces.  Even the ones that are boxes in the background.
        if len(faceboxes) == 0:
            # Didn't find any faces!
            text_loc = (30, 30)
            image = cv2.putText(image, "No face captured", text_loc, font, 
                               fontScale, color_all_faces, thickness, cv2.LINE_AA)
        else:
            for (x, y, width, height) in faceboxes:
                # Get the grayscale of the largest "face" as its own image
                cropped_face = grayimg[y:y + height, x:x + width]
                
                # Draw a box around just that face on the color image.
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                emotion_prediction = get_predicted_emotion(cropped_face)
            
                # Put text on the color image
                text_loc = (x + 5, y - 15)  # location on image
                image = cv2.putText(image, emotion_prediction, text_loc, font, 
                               fontScale, color_all_faces, thickness, cv2.LINE_AA)

    cv2.imshow("Emotion Capture, press x to exit", image)
    
    # User can request an exit by hitting X
    exit_wanted = ( cv2.waitKey(1) & 0xFF == ord('x') )
  
# Cleanup
vidcap.release()
cv2.destroyAllWindows()

