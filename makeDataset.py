video_path='C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\videos'
labels_path='C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\labels'

import os
import cv2
import tensorflow as tf
import numpy as np

from scipy.io import loadmat



def get_data(videos_path,label_path, sample_labels = True,fps = 15, denoising = False, blur = True, resize = True,
                normalize = False, frame_width = 256, frame_height=256, background_sub = False, object_trajectory = False, 
                data_from_numpy = False):
  
    data  , length= load_video(videos_path,fps = fps, denoising = denoising,blur = blur,resize = resize,normalize = normalize,frame_width = frame_width,
                             frame_height = frame_height,background_sub = background_sub,object_trajectory = object_trajectory,
                             data_from_numpy = data_from_numpy )
      # print('loaded ',path)
      # labels = np.load(labels_path,allow_pickle= True)
    labels = get_video_label(label_path,length,half = True,data_from_numpy=data_from_numpy)
    return data , labels


def load_video(video_path, fps = 15, denoising = False, blur = True, resize = True, normalize = False, frame_width = 256, frame_height=256, 
               background_sub = False,object_trajectory = False, data_from_numpy = False):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # taking the Background from 2nd second 
    cap.set(cv2.CAP_PROP_POS_MSEC,2*1000)
    reval, background = cap.read()
    # Preprocess the Background
    background = cv2.GaussianBlur(background, (3,3), 0) 
    # Get back to the 0 second
    cap.set(cv2.CAP_PROP_POS_MSEC,0)
    video=[]
    while True:
    #video.append(img)
        if object_trajectory: # If it's desired to Load video frames and get the object trajectory for each 6 consecutive frames
            chunk = []
            for i in range(0,6):
                reval, img = cap.read() # This is neglected because we load at 15 FPS , and video is stored at 30 FPS
                reval, img = cap.read()

                if not reval:
                    # End of frames
                    print('END')
                    video = np.array(video)
                    return video , length
                if background_sub:      # This is needed for the 4th Model, as we subtract the BG and then get OT
                    # Subtract Background
                    img = subtract_background(img,background)
                img = cv2.resize(img, (224, 224))
                chunk.append(img) # Here we store video chunk of 6 Frames and then pass it to get_stacked_pixel_trajectory
            chunk = np.array(chunk)
            video.append(get_stacked_pixel_trajectory(chunk,2))

        else:
            reval, img = cap.read() # Video is stored at 30 FPS, we want to load it at 15 FPS so we ignore a frame each iteration
            reval, img = cap.read() # Faster than cap.set()

            if not reval:
                # End of frames
                break
            if background_sub:
                # Subtract Background
                img = subtract_background(img,background)

            if resize:
                img = cv2.resize(img, (frame_width, frame_height)) 
            if blur:
                img = cv2.GaussianBlur(img, (21,21), 0) 
            if normalize:
                img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                  
            video.append(img)

    video = np.array(video)
        # try:
        #   video = video[[i for i in range(0,len(video),2)]]
        # except IndexError:
        # print('index error')
        # pass
    cap.release()
    del cap
    print(length)
    return video , length
    
def get_video_label(label_path,video_lenght, half = True, data_from_numpy = False):
    label_data = loadmat(label_path)
    # Initially each frame is label 0 
    label = [0 for i in range(video_lenght)] 
    label = np.array(label)

    for category_number in range(5):
        # Each video chunk in the same class defined as in labels
        for video_chunk in label_data['tlabs'][category_number][0]:
            label[video_chunk[0]:video_chunk[1]] = category_number+1

    label = np.array(label)
    if half:
        label = label[[i for i in range(1,video_lenght,2)]]

    return (label)


CLASS_POINTS = ['1_1','1_2','1_3','2_1','2_2','2_3','3_1','3_2','3_3','4_1','4_2','4_3','5_1','5_2','5_3','6_1','6_2','6_3']

dataStuff = []
labelsStuff = []

for classPoint in CLASS_POINTS:
    video_path='C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\videos\\'+ classPoint + '_crop.mp4'
    label_path='C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\labels\\'+ classPoint +'_label.mat'
    x=loadmat(label_path)
    data,labels=get_data(video_path,label_path)
    dataStuff.append(data)
    labelsStuff.append(labels)


fOne = open("C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\dataStuff.npy", "w+")
fTwo = open("C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\dataset\\labelStuff.npy", "w+")
np.save(fOne, dataStuff)
np.save(fTwo, labelsStuff)





