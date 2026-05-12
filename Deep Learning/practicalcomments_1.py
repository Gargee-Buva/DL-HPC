# ==========================================================
# PRACTICAL 1 : Boston Housing Price Prediction
# using Linear Regression
# ==========================================================

# drive.mount('/content/drive') is used to connect Google Drive with Google Colab so that files and datasets stored in Drive can be accessed inside the notebook.

# ----------------------------------------------------------
# STEP 1 : Import Required Libraries
# ----------------------------------------------------------

# numpy for numerical operations
import numpy as np

# pandas for dataframe handling
import pandas as pd

#Keras is used to build and train deep learning models.
from tensorflow import keras

#StandardScaler is used to normalize input features so that all values are in a similar range for efficient neural network training.
from sklearn.preprocessing import StandardScaler

#Sequential is used to create a neural network where layers are arranged sequentially.
from tensorflow.keras.models import Sequential

# Dense means : Every neuron connects to every neuron of next layer.
#Dense layers:
#1. learn relationships between features and house prices
#2. perform regression prediction
from tensorflow.keras.layers import Dense

#RMSprop is an optimizer used to minimize loss and improve neural network learning by updating weights efficiently.
from tensorflow.keras.optimizers import RMSprop


# ----------------------------------------------------------
# STEP 2 : Load Boston Housing Dataset
# ----------------------------------------------------------

## (Dataset Available directly in Keras)

# keras.datasets.boston_housing.load_data()
# loads the Boston Housing dataset

# X_train -> input features used for training
# Y_train -> output prices used for training

# X_test -> input features used for testing
# Y_test -> actual prices used for testing

# The dataset is automatically divided into:
# training data and testing data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

# (number_of_training_samples, number_of_features)

# Example:
# (404, 13)

# Meaning:
# 404 house records for training
# each house has 13 features

print("Training data shape:", X_train.shape)
# X_test.shape shows:
# number of testing samples and features
print("Test data shape:", X_test.shape)  
print("Train output data shape:", Y_train.shape) # number of output labels/prices

# ----------------------------------------------------------
# STEP 3 : Feature Scaling using StandardScaler
# ----------------------------------------------------------
# Create StandardScaler object
# Used to normalize data into similar range

scaler = StandardScaler()


# fit_transform() does two things:
# 1. Learns mean and standard deviation from training data
# 2. Scales training data

# Why?
# Neural networks perform better when all features
# are in similar range

X_train = scaler.fit_transform(X_train)


# transform() scales testing data
# using SAME mean and standard deviation
# learned from training data

# Important:
# We never fit scaler separately on test data
# Use same scaling parameters from training data
# to avoid data leakage

X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# STEP 5 : Create histogram plot of house prices
# ----------------------------------------------------------

import seaborn as sns
# seaborn is a data visualization library
# used for creating attractive statistical graphs
import matplotlib.pyplot as plt
# matplotlib.pyplot is used for plotting graphs and visualizations
sns.histplot(Y_train, kde=True)
#histplot() → creates histogram graph
#Y_train → training house prices data
#kde=True → adds smooth distribution curve on graph
plt.xlabel("House Price ($1000s)")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()
# Graph Shows how house prices are distributed in training data

# ----------------------------------------------------------
# STEP 6 : Boxplot building
# ----------------------------------------------------------

#| Histogram                    | Boxplot                 |
#| ---------------------------- | ----------------------- |
#| Shows frequency distribution | Shows spread & outliers |
#| Detailed distribution        | Statistical summary     |


# ----------------------------------------------------------
# STEP 7 : dataframes arrangement
# ----------------------------------------------------------

# Import pandas library for dataframe handling
import pandas as pd

# Convert training data into pandas DataFrame
# for easier analysis and visualization
df = pd.DataFrame(X_train)

# Assign feature names to dataframe columns
df.columns = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT'
]


# Add house price column to dataframe
df['PRICE'] = Y_train


#Calculates correlation between every pair of columns
#Checks how strongly features are related
correlation = df.corr()

#Extracts only correlations with house price
#Shows which features affect price most
correlation['PRICE']

# ----------------------------------------------------------
# STEP 8 : Heatmap display
# ----------------------------------------------------------

# Create figure with larger size for heatmap display
fig,axes = plt.subplots(figsize=(15,12))


# Plot heatmap of correlation matrix
# square=True makes cells square
# annot=True displays correlation values inside cells

sns.heatmap(correlation, square=True, annot=True)

##Purpose of heatmap :-

#visually show correlation between features
#identify strongly related features
#understand how features affect house price

#What Does Heatmap Tell?
#Dark/high values → strong correlation
#Light/low values → weak correlation
#Positive values → direct relationship
#Negative values → inverse relationship

# ----------------------------------------------------------
# STEP 9 : Predict House Prices
# ----------------------------------------------------------

# Create figure with width=20 and height=5
plt.figure(figsize = (20,5))


# Select features to compare with house prices
features = ['LSTAT','RM','PTRATIO']


# Loop through each feature
for i, col in enumerate(features):

    # Create subplot for each feature
    plt.subplot(1, len(features), i+1)

    # x-axis contains selected feature values
    x = df[col]

    # y-axis contains house prices
    y = df.PRICE

    # Plot scatter graph between feature and price
    plt.scatter(x, y, marker='o')

    # Set graph title
    plt.title("Variation in House prices")

    # Set x-axis label
    plt.xlabel(col)

    # Set y-axis label
    plt.ylabel('"House prices in $1000"')

    #These scatter plots are used to:
    #visualize relationship between features and house prices
    #observe trends/patterns in data

# ----------------------------------------------------------
# STEP 10 : Build Deep Neural Network Model
# ----------------------------------------------------------

# Create Sequential Neural Network model
model = Sequential()


# Add first hidden layer with:
# 128 neurons
# ReLU activation function
# input_shape defines number of input features

model.add(Dense(128,
                activation='relu',
                input_shape=X_train[0].shape))


# Add second hidden layer with 64 neurons
# Learns deeper patterns from data

model.add(Dense(64,
                activation='relu'))


# Add third hidden layer with 32 neurons
# Further refines learned features

model.add(Dense(32,
                activation='relu'))


# Output layer with 1 neuron
# Predicts single continuous value (house price)

model.add(Dense(1))


# Display model architecture summary
# Shows layers, neurons, and parameters

model.summary()


# ----------------------------------------------------------
# Compile Model
# ----------------------------------------------------------

# loss='mse'
# Mean Squared Error used for regression

# optimizer='rmsprop'
# Optimizer updates weights to reduce loss

# metrics=['mae']
# Mean Absolute Error used for evaluation

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mae'])


# ----------------------------------------------------------
# Train Deep Neural Network
# ----------------------------------------------------------

# fit() trains the model

# epochs=50
# complete passes through training data

# batch_size=1
# one sample processed at a time

# verbose=1
# displays training progress

# validation_data
# evaluates performance on test data

history = model.fit(X_train,
                    Y_train,
                    epochs=50,
                    batch_size=1,
                    verbose=1,
                    validation_data=(X_test, Y_test))

# What Actually Happens Here?

# The neural network:

# Takes house features as input
# Passes data through hidden layers
# Learns patterns affecting house prices
# Predicts house price
# Calculates error using MSE
# Updates weights using RMSprop
# Repeats for 50 epochs to improve accuracy

# ----------------------------------------------------------
# STEP 11 : Calculate Error
# ----------------------------------------------------------
# Predict house prices using test data
y_pred = model.predict(X_test)


# Evaluate model performance on test data
# Returns:
# mse_nn -> Mean Squared Error
# mae_nn -> Mean Absolute Error

mse_nn, mae_nn = model.evaluate(X_test, Y_test)


# Display Mean Squared Error
# Lower value means better prediction accuracy

print('Mean squared error on test data: ', mse_nn)


# Display Mean Absolute Error
# Shows average prediction error

print('Mean absolute error on test data: ', mae_nn)


# Display predicted house prices
y_pred

# predict() → predicts house prices
# evaluate() → checks model performance
# mse_nn → squared prediction error
# mae_nn → average prediction error
# y_pred → predicted house prices array