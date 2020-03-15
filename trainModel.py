import matplotlib

import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers.core import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import numpy as np
import argparse
import pickle
import cv2
import os




LABELS = set([ "ReachToShelf", "RetractFromShelf", "HandInShelf", "InspectProduct", "InspectShelf" ])

print("Loading images...............")
imagePaths=list(paths.list_images('C:\\Users\\Hepsiba Navneetha\\Desktop\\MotionDetection\\TcsProject'))

data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    if label not in LABELS:
        continue
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    data.append(image)
    labels.append(label)

lb = LabelBinarizer()
labels =lb.fit_transform(labels)

(trainX , testX , trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

trainX = np.array(trainX)
testX = np.array(testX)

print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
datagen.mean = mean

baseModel = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7), padding='same')(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
#headModel = Flatten(name="flatten")(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
epochs=100
opt = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-4 /(epochs))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

early_stop = EarlyStopping(monitor='loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint('sai.h5',
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min', 
                             period=1)


#genVal = trainAug.flow(trainX, train_y, batch_size=32) 

print("[INFO] training head...")

H = model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
                    steps_per_epoch=len(trainX) / 32, epochs=epochs, callbacks = [early_stop, checkpoint])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

N = epochs
print("[INFO] serializing network...")
model.save("model.h5")
f = open("label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()

