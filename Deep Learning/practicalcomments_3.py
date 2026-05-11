# ==========================================
# PRACTICAL 3 : Fashion MNIST using CNN
# ==========================================

# ------------------------------------------------
# STEP 1 : Import Required Libraries
# ------------------------------------------------

# numpy is used for numerical operations and array handling
import numpy as np

# matplotlib is used for plotting and displaying images
import matplotlib.pyplot as plt

# tensorflow is the main deep learning framework
import tensorflow as tf

# keras contains high-level APIs for building neural networks
from tensorflow import keras

# Fashion MNIST dataset is imported from keras datasets
from keras.datasets import fashion_mnist

# Sequential is used to create neural networks layer-by-layer
from keras.models import Sequential

# Import CNN related layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Used for converting labels into one-hot encoded vectors
from tensorflow.keras.utils import to_categorical


# ------------------------------------------------
# STEP 2 : Load Fashion-MNIST Dataset
# ------------------------------------------------

# load_data() returns:
# trainX -> training images
# trainY -> training labels
# testX  -> testing images
# testY  -> testing labels

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()


# ------------------------------------------------
# STEP 3 : Check Shape of Dataset
# ------------------------------------------------

# Shape tells:
# (number_of_images, image_height, image_width)

print("Training Data Shape :", trainX.shape)
print("Testing Data Shape :", testX.shape)

# Expected:
# (60000, 28, 28)
# meaning:
# 60000 grayscale images
# each image is 28x28 pixels


# ------------------------------------------------
# STEP 4 : Display Sample Images
# ------------------------------------------------

# Plot first 9 images from training dataset

for i in range(9):

    # create subplot grid (3 rows, 3 columns)
    plt.subplot(330 + 1 + i)

    # display image using grayscale color map
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

# show all images
plt.show()


# ------------------------------------------------
# STEP 5 : Reshape Dataset for CNN
# ------------------------------------------------

# CNN expects 4D input:
# (samples, height, width, channels)

# Here:
# height = 28
# width  = 28
# channels = 1 because image is grayscale

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))


# ------------------------------------------------
# STEP 6 : Normalize Pixel Values
# ------------------------------------------------

# Original pixel values range:
# 0 to 255

# Dividing by 255 converts values into:
# 0 to 1

# This improves:
# - training speed
# - convergence
# - model stability

trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0


# ------------------------------------------------
# STEP 7 : One-Hot Encode Labels
# ------------------------------------------------

# Labels are integers from 0 to 9

# Example:
# 3 becomes:
# [0 0 0 1 0 0 0 0 0 0]

# Neural networks work better with categorical vectors

trainY = to_categorical(trainY)
testY = to_categorical(testY)


# ------------------------------------------------
# STEP 8 : Build CNN Model
# ------------------------------------------------

# Sequential model means layers are added one after another

model = Sequential()


# ------------------------------------------------
# CONVOLUTION LAYER
# ------------------------------------------------

# Conv2D:
# 32 -> number of filters/kernels
# (3,3) -> filter size

# Purpose:
# Detect image features like:
# - edges
# - textures
# - patterns

# activation='relu'
# ReLU introduces nonlinearity

model.add(Conv2D(32,
                 (3,3),
                 activation='relu',
                 input_shape=(28,28,1)))


# ------------------------------------------------
# MAX POOLING LAYER
# ------------------------------------------------

# Pool size = (2,2)

# Purpose:
# Reduce image dimensions
# Reduce computation
# Preserve important features

# MaxPooling selects maximum value from region

model.add(MaxPooling2D((2,2)))


# ------------------------------------------------
# FLATTEN LAYER
# ------------------------------------------------

# CNN output is multidimensional

# Flatten converts:
# 2D feature maps
# into
# 1D vector

# Needed before Dense layers

model.add(Flatten())


# ------------------------------------------------
# DENSE / FULLY CONNECTED LAYER
# ------------------------------------------------

# Dense(100):
# 100 neurons

# Learns complex relationships between extracted features

model.add(Dense(100, activation='relu'))


# ------------------------------------------------
# OUTPUT LAYER
# ------------------------------------------------

# Dense(10):
# 10 neurons because dataset has 10 classes

# softmax activation:
# Converts outputs into probabilities

# Highest probability becomes predicted class

model.add(Dense(10, activation='softmax'))


# ------------------------------------------------
# STEP 9 : Compile Model
# ------------------------------------------------

# optimizer='adam'
# Adam optimizer updates weights efficiently

# loss='categorical_crossentropy'
# Used for multiclass classification

# metrics=['accuracy']
# Accuracy will be displayed during training

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ------------------------------------------------
# STEP 10 : Train CNN Model
# ------------------------------------------------

# fit() trains the neural network

# epochs=10
# complete passes through training dataset

# batch_size=32
# 32 images processed together

history = model.fit(trainX,
                    trainY,
                    epochs=10,
                    batch_size=32,
                    validation_data=(testX, testY))


# ------------------------------------------------
# STEP 11 : Evaluate Model
# ------------------------------------------------

# evaluate() checks model performance on unseen data

loss, accuracy = model.evaluate(testX, testY)

print("Test Accuracy :", accuracy)


# ------------------------------------------------
# STEP 12 : Predict Classes
# ------------------------------------------------

# predict() gives probabilities for each class

predictions = model.predict(testX)

# np.argmax() returns index of highest probability

predicted_class = np.argmax(predictions[0])

print("Predicted Class :", predicted_class)


# ------------------------------------------------
# STEP 13 : Display Test Image
# ------------------------------------------------

# Display image to verify prediction

plt.imshow(testX[0].reshape(28,28), cmap='gray')

plt.title("Predicted Class : {}".format(predicted_class))

plt.show()