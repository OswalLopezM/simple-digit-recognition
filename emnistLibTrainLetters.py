# # Visualization Dependencies
# from IPython.display import Image, SVG
# import seaborn as sns

# # Filepaths, Numpy, Tensorflow
# import os
# import numpy as np
# import tensorflow as tf

# # Keras
# from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras import backend as K
# from tensorflow.keras.datasets import mnist
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

from emnist import extract_training_samples
images_train, labels_train = extract_training_samples('balanced')
from emnist import extract_test_samples
images_test, labels_test = extract_test_samples('balanced')


dims = images_train.shape[1] * images_train.shape[2]

## DENSE NN
X_train = images_train.reshape(images_train.shape[0], dims)
X_test = images_test.reshape(images_test.shape[0], dims)

## CONV NN
# X_train = images_train.reshape(images_train.shape[0], 28,28,1)
# X_test = images_test.reshape(images_test.shape[0], 28,28,1)


print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# One-Hot Encoding

from keras.utils import np_utils # used to convert array of labeled data to one-hot vector
# should be 26 but out of index?
# Effects accuracy as have a class where their will be no results
num_classes = 47
y_train = np_utils.to_categorical(labels_train, num_classes)
y_test = np_utils.to_categorical(labels_test, num_classes)


# Empty Sequential model
from tensorflow.keras.models import Sequential
model = Sequential()

#Layers
## DENSE NN
# 1 - number of elements (pixels) in each image
# Dense layer - when every node from previous layer is connected to each node in current layer
model.add(Dense(500, activation='relu'))

# Second Hidden Layer
model.add(Dense(500, activation='relu'))

# Output Layer - number of nodes corresponds to number of y labels
model.add(Dense(num_classes, activation='softmax'))

## CONV NN
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop' , metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, batch_size=128, epochs=10, shuffle=True, verbose=2)

# Save Model

#SAVE DENSE
model.save("emnist_trained_dense.h5")

# #SAVE CNN
# model.save("emnist_trained.h5")
