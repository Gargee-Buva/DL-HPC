# ==========================================================
# PRACTICAL 3 : Fashion MNIST Classification using CNN
# ==========================================================

# ----------------------------------------------------------
# STEP 1 : Import Required Libraries
# ----------------------------------------------------------

# numpy is used for numerical operations and arrays
import numpy as np 

# matplotlib is used for plotting graphs and displaying images
import matplotlib.pyplot as plt 

# Fashion-MNIST dataset loader
from tensorflow.keras.datasets import fashion_mnist

# Sequential is used to build neural network layer-by-layer
from tensorflow.keras.models import Sequential

# CNN layers
# Conv2D -> feature extraction
# MaxPooling2D -> reduce dimensions
# Dense -> fully connected layer
# Flatten -> converts 2D into 1D
# Dropout -> reduces overfitting

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# to_categorical converts labels into one-hot encoded vectors
from tensorflow.keras.utils import to_categorical 


# ----------------------------------------------------------
# STEP 2 : Load Fashion-MNIST Dataset
# ----------------------------------------------------------

# load_data() returns:
# trainX -> training images
# trainY -> training labels
# testX -> testing images
# testY -> testing labels

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

# Display dataset shape

# trainX.shape:
# (60000, 28, 28)

# Meaning:
# 60000 grayscale images
# each image size = 28x28 pixels

print(trainX.shape, testX.shape)


# ----------------------------------------------------------
# STEP 3 : Reshape Images for CNN
# ----------------------------------------------------------

# CNN expects 4D input:
# (samples, height, width, channels)

# channels = 1 because images are grayscale

trainX = trainX.reshape((60000, 28, 28, 1))
testX = testX.reshape((10000, 28, 28, 1))


# ----------------------------------------------------------
# STEP 4 : Normalize Pixel Values
# ----------------------------------------------------------

# Original pixel values:
# 0 to 255

# Divide by 255 to convert values into:
# 0 to 1

# Benefits:
# faster training
# stable learning
# improved convergence

trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0


# ----------------------------------------------------------
# STEP 5 : One-Hot Encode Labels
# ----------------------------------------------------------

# Convert labels into binary vectors

# Example:
# class 3 becomes:
# [0 0 0 1 0 0 0 0 0 0]

# Neural networks work better with categorical vectors

from tensorflow.keras.utils import to_categorical

trainY = to_categorical(trainY)
testY = to_categorical(testY)


# ----------------------------------------------------------
# STEP 6 : Build CNN Model
# ----------------------------------------------------------

# Sequential model:
# layers added one after another

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout

model = Sequential()


# ----------------------------------------------------------
# Convolution Block 1
# ----------------------------------------------------------

# Conv2D:
# 32 filters/kernels
# (3,3) filter size

# Purpose:
# extract image features like:
# edges, textures, shapes

# activation='relu'
# introduces nonlinearity

# input_shape=(28,28,1)
# image size = 28x28 grayscale

model.add(Conv2D(32, (3,3),
                 activation='relu',
                 input_shape=(28,28,1)))


# MaxPooling2D:
# reduces image dimensions
# keeps important features
# reduces computation

model.add(MaxPooling2D((2,2)))


# ----------------------------------------------------------
# Convolution Block 2
# ----------------------------------------------------------

# Second convolution layer:
# learns deeper image patterns

model.add(Conv2D(64, (3,3),
                 activation='relu'))

# Further reduce dimensions

model.add(MaxPooling2D((2,2)))


# ----------------------------------------------------------
# Flatten Layer
# ----------------------------------------------------------

# Converts 2D feature maps into 1D vector
# before passing to dense layer

model.add(Flatten())


# ----------------------------------------------------------
# Fully Connected Layer
# ----------------------------------------------------------

# Dense(128):
# 128 neurons

# Learns complex relationships
# from extracted image features

model.add(Dense(128, activation='relu'))


# ----------------------------------------------------------
# Dropout Layer
# ----------------------------------------------------------

# Dropout randomly disables neurons during training

# Purpose:
# reduce overfitting
# improve generalization

model.add(Dropout(0.5))


# ----------------------------------------------------------
# Output Layer
# ----------------------------------------------------------

# Dense(10):
# 10 neurons because:
# Fashion-MNIST has 10 classes

# softmax activation:
# converts outputs into probabilities

# Highest probability becomes predicted class

model.add(Dense(10, activation='softmax'))


# ----------------------------------------------------------
# STEP 7 : Compile CNN Model
# ----------------------------------------------------------

model.compile(

    # Adam optimizer updates weights efficiently
    optimizer='adam',

    # categorical_crossentropy used for
    # multiclass classification

    loss='categorical_crossentropy',

    # accuracy used for performance evaluation
    metrics=['accuracy']
)


# ----------------------------------------------------------
# What model.compile() Actually Does
# ----------------------------------------------------------

# optimizer:
# improves learning by updating weights

# loss:
# calculates prediction error

# metrics:
# measures model performance


# ----------------------------------------------------------
# STEP 8 : Train CNN Model
# ----------------------------------------------------------

history = model.fit(

    # Training images
    trainX,

    # Training labels
    trainY,

    # epochs=10
    # complete passes through dataset
    epochs=10,

    # batch_size=32
    # 32 images processed together
    batch_size=32,

    # validation_data:
    # checks performance on unseen data
    validation_data=(testX, testY)
)


# ----------------------------------------------------------
# What Happens During Training?
# ----------------------------------------------------------

# CNN:

# 1. Reads image pixels
# 2. Extracts image features using convolution
# 3. Reduces dimensions using pooling
# 4. Learns patterns from clothes
# 5. Predicts clothing category
# 6. Calculates error
# 7. Updates weights using Adam optimizer
# 8. Repeats for multiple epochs


# ----------------------------------------------------------
# STEP 9 : Evaluate Model
# ----------------------------------------------------------

# evaluate() checks model performance
# on testing data

loss, accuracy = model.evaluate(testX, testY)

# Display final test accuracy

print("Test Accuracy:", accuracy)


# ----------------------------------------------------------
# Meaning of Final Output
# ----------------------------------------------------------

# loss:
# prediction error on testing data

# accuracy:
# percentage of correct predictions

# Higher accuracy means:
# better image classification performance

#------------------------------------------------------------

# Overview :-
# Load Fashion-MNIST dataset
# Reshape images for CNN input
# Normalize pixel values
# Convert labels into one-hot vectors
# Build CNN model
# Add convolution and pooling layers
# Flatten feature maps
# Add dense/output layers
# Compile CNN model
# Train model using training images
# Evaluate accuracy on test data
# Predict clothing image classes