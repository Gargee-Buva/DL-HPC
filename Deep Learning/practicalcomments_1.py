# ==========================================================
# PRACTICAL 1 : Boston Housing Price Prediction
# using Linear Regression
# ==========================================================

# ----------------------------------------------------------
# STEP 1 : Import Required Libraries
# ----------------------------------------------------------

# numpy for numerical operations
import numpy as np

# pandas for dataframe handling
import pandas as pd

# matplotlib for plotting graphs
import matplotlib.pyplot as plt

# sklearn contains machine learning algorithms
from sklearn.datasets import load_boston

# train_test_split used to divide dataset
from sklearn.model_selection import train_test_split

# LinearRegression algorithm
from sklearn.linear_model import LinearRegression

# Mean Squared Error calculation
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------
# STEP 2 : Load Boston Housing Dataset
# ----------------------------------------------------------

# load_boston() loads housing price dataset

boston = load_boston()


# ----------------------------------------------------------
# STEP 3 : Convert Dataset into DataFrame
# ----------------------------------------------------------

# Convert dataset into tabular format

data = pd.DataFrame(boston.data)

# Assign column names
data.columns = boston.feature_names

# Add target column named Price
data['Price'] = boston.target


# ----------------------------------------------------------
# STEP 4 : Display Dataset
# ----------------------------------------------------------

print(data.head())


# ----------------------------------------------------------
# STEP 5 : Separate Input and Output Data
# ----------------------------------------------------------

# x contains input features
# Example:
# crime rate, rooms, tax etc.

x = boston.data

# y contains output values
# house prices

y = boston.target


# ----------------------------------------------------------
# STEP 6 : Split Dataset into Training and Testing
# ----------------------------------------------------------

# test_size=0.2
# 20% data used for testing

# random_state=0
# ensures same random split every time

xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)


# ----------------------------------------------------------
# STEP 7 : Create Linear Regression Model
# ----------------------------------------------------------

# Create Linear Regression object

regressor = LinearRegression()


# ----------------------------------------------------------
# STEP 8 : Train Model
# ----------------------------------------------------------

# fit() trains the model

# Model learns relationship between:
# input features and house prices

regressor.fit(xtrain, ytrain)


# ----------------------------------------------------------
# STEP 9 : Predict House Prices
# ----------------------------------------------------------

# predict() predicts prices using testing data

y_pred = regressor.predict(xtest)


# ----------------------------------------------------------
# STEP 10 : Plot Graph
# ----------------------------------------------------------

# Scatter plot compares:
# actual prices vs predicted prices

plt.scatter(ytest,
            y_pred,
            c='green')

# x-axis label
plt.xlabel("Actual Price")

# y-axis label
plt.ylabel("Predicted Price")

# graph title
plt.title("Actual vs Predicted House Prices")

# display graph
plt.show()


# ----------------------------------------------------------
# STEP 11 : Calculate Error
# ----------------------------------------------------------

# Mean Squared Error measures prediction error

mse = mean_squared_error(ytest, y_pred)

print("Mean Squared Error :", mse)


# ----------------------------------------------------------
# STEP 12 : Display Sample Predictions
# ----------------------------------------------------------

print("Actual Price :", ytest[0])

print("Predicted Price :", y_pred[0])


# ----------------------------------------------------------
# Understanding What Model is Doing
# ----------------------------------------------------------

# The model tries to:
# 1. Learn relationship between input features and price
# 2. Find best fit regression line
# 3. Minimize prediction error
# 4. Predict prices for unseen houses

# Lower MSE means:
# better prediction accuracy