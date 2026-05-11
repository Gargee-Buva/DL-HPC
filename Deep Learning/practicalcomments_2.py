# ==========================================================
# PRACTICAL 2 : IMDB Movie Review Sentiment Analysis using DNN
# ==========================================================

# ----------------------------------------------------------
# STEP 1 : Import Required Libraries
# ----------------------------------------------------------

# numpy is used for numerical operations
import numpy as np

# keras provides deep learning APIs
from tensorflow import keras

# Sequential is used to create neural network layer-by-layer
from keras.models import Sequential

# Dense creates fully connected neural network layers
from keras.layers import Dense

# IMDB dataset contains movie reviews
from keras.datasets import imdb


# ----------------------------------------------------------
# STEP 2 : Load IMDB Dataset
# ----------------------------------------------------------

# num_words=10000 means:
# only top 10,000 frequently occurring words are used

# x_train -> movie reviews for training
# y_train -> labels for training
# x_test  -> movie reviews for testing
# y_test  -> labels for testing

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)


# ----------------------------------------------------------
# STEP 3 : Understand Dataset
# ----------------------------------------------------------

# Each review is stored as integer sequence

# Example:
# [1, 14, 20, 45]

# Each integer represents a word index

print("Sample Review :", x_train[0])

print("Label :", y_train[0])

# Label meanings:
# 1 -> Positive Review
# 0 -> Negative Review


# ----------------------------------------------------------
# STEP 4 : Vectorization / One-Hot Encoding
# ----------------------------------------------------------

# Neural networks cannot understand raw text

# So we convert reviews into vectors of:
# 0s and 1s

# 1 -> word exists
# 0 -> word absent

# We create a function for vectorization

def vectorize_sequences(sequences, dimension=10000):

    # Create matrix filled with zeros
    # rows = number of reviews
    # columns = 10000 words

    results = np.zeros((len(sequences), dimension))

    # enumerate() gives:
    # i -> index
    # sequence -> actual review

    for i, sequence in enumerate(sequences):

        # Mark positions of existing words as 1
        results[i, sequence] = 1.

    return results


# Convert training reviews into vectors
x_train = vectorize_sequences(x_train)

# Convert testing reviews into vectors
x_test = vectorize_sequences(x_test)


# ----------------------------------------------------------
# STEP 5 : Convert Labels into Float
# ----------------------------------------------------------

# Convert labels into float32 format

y_train = np.asarray(y_train).astype('float32')

y_test = np.asarray(y_test).astype('float32')


# ----------------------------------------------------------
# STEP 6 : Build Deep Neural Network
# ----------------------------------------------------------

# Sequential model:
# layers added one after another

model = Sequential()


# ----------------------------------------------------------
# FIRST HIDDEN LAYER
# ----------------------------------------------------------

# Dense(16):
# 16 neurons

# activation='relu'
# ReLU introduces nonlinearity

# input_shape=(10000,)
# input vector contains 10000 features

model.add(Dense(16,
                activation='relu',
                input_shape=(10000,)))


# ----------------------------------------------------------
# SECOND HIDDEN LAYER
# ----------------------------------------------------------

# Another hidden layer helps learn deeper patterns

model.add(Dense(16,
                activation='relu'))


# ----------------------------------------------------------
# OUTPUT LAYER
# ----------------------------------------------------------

# Dense(1):
# single output neuron because:
# binary classification

# sigmoid activation:
# output probability between 0 and 1

model.add(Dense(1,
                activation='sigmoid'))


# ----------------------------------------------------------
# STEP 7 : Compile Model
# ----------------------------------------------------------

# optimizer='rmsprop'
# optimizer updates weights during training

# loss='binary_crossentropy'
# used for binary classification

# metrics=['accuracy']
# accuracy displayed during training

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ----------------------------------------------------------
# STEP 8 : Create Validation Dataset
# ----------------------------------------------------------

# First 10,000 reviews used for validation

x_val = x_train[:10000]

partial_x_train = x_train[10000:]

y_val = y_train[:10000]

partial_y_train = y_train[10000:]


# ----------------------------------------------------------
# STEP 9 : Train Deep Neural Network
# ----------------------------------------------------------

# epochs=20
# complete passes through dataset

# batch_size=512
# 512 reviews processed together

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# ----------------------------------------------------------
# STEP 10 : Evaluate Model
# ----------------------------------------------------------

# evaluate() checks performance on test dataset

results = model.evaluate(x_test, y_test)

print("Test Loss :", results[0])

print("Test Accuracy :", results[1])


# ----------------------------------------------------------
# STEP 11 : Predict Sentiment
# ----------------------------------------------------------

# predict() gives probability values

predictions = model.predict(x_test)

print(predictions[0])

# If prediction close to 1:
# Positive Review

# If prediction close to 0:
# Negative Review