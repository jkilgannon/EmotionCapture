"""
 
CISC 642
Final Project
Test the CNNs+ for emotional detection
Jon Kilgannon
 
"""

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import os
import numpy as np

data_location = "D:\\!data\\college\\2021Spring\\fer2103\\test\\"

# The text versions of the predicted emotions; used for on-screen display and for directory names
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# List all the models in the directory with this program
print("Loading models...")
emotion_nets = []
net_names = []
for file in os.listdir("."):
    if file.endswith(".h5"):
        print(file)
        net_names.append(file)
        emotion_net = keras.models.load_model(file)
        emotion_nets.append(emotion_net)
        
print("Number of models: " + str(len(emotion_nets)))

def get_predicted_emotion(face):
    all_predictions = []
    
    for e_n in emotion_nets:
        # Let each neural net predict
        pred = e_n.predict(npary)
        emotion_prediction = np.argmax(pred, axis=-1)
        emotion_prediction = emotion_prediction[0]          # It's returned in a one-element array
        all_predictions.append(int(emotion_prediction))

    # Vote!
    prediction_counts = [0,0,0,0,0,0,0]
    for prediction in all_predictions:
        prediction_counts[prediction] = prediction_counts[prediction] + 1 
        
    #print(str(all_predictions) + "   " + str(prediction_counts))
    
    if prediction_counts.count(1) == len(prediction_counts):
        # Everyone voted differently.  There's no consensus
        max_votes = -1
    else:
        max_votes = np.argmax(prediction_counts)

    return max_votes



# Load all the images into memory so this runs faster
print("Loading images...")
all_images = []
for emotion_name in emotions:
    emotion_images = []
    dir_loc = data_location + emotion_name + "\\"
    
    for file in os.listdir(dir_loc):
        # Predict for each file in this directory
        new_face = cv2.imread(dir_loc + file, cv2.IMREAD_GRAYSCALE)
        
        resize = cv2.resize(new_face, (48,48), interpolation = cv2.INTER_AREA)
        npary = np.array(resize)
        npary = npary.reshape((1,48,48,1))      # Make it into a one-item batch for the network to digest
        
        emotion_images.append(npary)
        
    all_images.append(emotion_images)

print("=====")
print("Images in each emotion:")
for a in all_images:
    print(len(a))
print("=====")

# For each image: Predict with all models.
for i in range(len(emotions)):
    correct_count = 0
    
    emotion_name = emotions[i]
    print("testing " + emotion_name + ", which is number " + str(i))
    
    for npary in all_images[i]:
        pred = get_predicted_emotion(npary)
        
        if i == pred:
            correct_count = correct_count + 1

    perc_correct = 100.0 * float(correct_count) / float(len(all_images[i]))

    print("file count: " + str(len(all_images[i])))
    print("correct predictions: " + str(correct_count))
    print("percentage correct: " + str(perc_correct))
    print("---------------------------------")

print("Done")

### The version below was used to capture the values of individual models
"""
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import os
import numpy as np

data_location = "D:\\!data\\college\\2021Spring\\fer2103\\test\\"

# The text versions of the predicted emotions; used for on-screen display and for directory names
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# List all the models in the directory with this program
emotion_nets = []
net_names = []
for file in os.listdir("."):
    if file.endswith(".h5"):
        net_names.append(file)

# Load all the images into memory so this runs faster
print("Loading images...")
all_images = []
for emotion_name in emotions:
    emotion_images = []
    dir_loc = data_location + emotion_name + "\\"
    
    for file in os.listdir(dir_loc):
        # Predict for each file in this directory
        new_face = cv2.imread(dir_loc + file, cv2.IMREAD_GRAYSCALE)
        
        resize = cv2.resize(new_face, (48,48), interpolation = cv2.INTER_AREA)
        npary = np.array(resize)
        npary = npary.reshape((1,48,48,1))      # Make it into a one-item batch for the network to digest
        
        emotion_images.append(npary)
        
    all_images.append(emotion_images)

print("=====")
print("Images in each emotion:")
for a in all_images:
    print(len(a))
print("=====")

# For each network: Predict for each image, and determine if it is correctly predicted
for net in net_names:
    print("Opening Emotion Net '" + net + "'")
    
    # Load the model
    emotion_net = keras.models.load_model(net)

    # Now open all of the validation images, and determine how well this model predicts them.
    correct_count = [0,0,0,0,0,0,0]     # Number of files predicted correctly w this label

    for i in range(len(emotions)):
        emotion_name = emotions[i]
        print("testing " + emotion_name + ", which is number " + str(i))
        
        dir_loc = data_location + emotion_name + "\\"
        
        for npary in all_images[i]:
            pred = emotion_net.predict(npary)
            emotion_prediction = np.argmax(pred, axis=-1)
            emotion_prediction = emotion_prediction[0]          # It's returned in a one-element array
            
            if i == emotion_prediction:
                correct_count[i] = correct_count[i] + 1
    
    perc_correct = [0,0,0,0,0,0,0]
    for i in range(len(perc_correct)):
        perc_correct[i] = 100.0 * float(correct_count[i]) / float(len(all_images[i]))
    
    print("file count: " + str(len(all_images[i])))
    print("correct predictions: " + str(correct_count))
    print("percentage correct: " + str(perc_correct))
    print("---------------------------------")

print("Done")
"""
