import tensorflow as tf
import cv2
import os
from tensorflow.python.keras.models import load_model
import numpy as np

sess = tf.InteractiveSession()

LABELS = [ "ReachToShelf", "RetractFromShelf", "HandInShelf", "InspectProduct", "InspectShelf" ]
model = load_model("C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\model.h5",compile=True)

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
        success, image = vidObj.read()
        pint = cv2.resize(image, (224,224))
        apple = np.array(pint)
        apple = apple.reshape((-1, 224,224,3))
        pred = model.predict(apple)
        print(LABELS[tf.argmax(pred[0], axis=0).eval()])
        cv2.imshow('image', image)
        cv2.waitKey(1)
  
FrameCapture(r"C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\1_1_crop.mp4") 

