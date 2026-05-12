# ==========================================================
# PRACTICAL 2 : Letter Recognition using Deep Neural Network
# ==========================================================

# ----------------------------------------------------------
# STEP 1 : Import Required Libraries
# ----------------------------------------------------------

# pandas is used for dataframe handling
import pandas as pd

# train_test_split is used to divide dataset
# into training and testing data
from sklearn.model_selection import train_test_split

# StandardScaler is used for feature scaling
# LabelBinarizer converts categorical labels into binary vectors
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# tensorflow is deep learning framework
import tensorflow as tf

# Sequential is used to create neural network
# layer-by-layer
from tensorflow.keras.models import Sequential

# Dense creates fully connected neural network layers
from tensorflow.keras.layers import Dense

# Adam optimizer improves model learning
# by updating weights efficiently
from tensorflow.keras.optimizers import Adam


# ----------------------------------------------------------
# STEP 2 : Load Dataset
# ----------------------------------------------------------

# Dataset URL from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

# Create column names:
# 1 target column -> "letter"
# 16 feature columns -> feat_0 to feat_15

col_names = ["letter"] + [f"feat_{i}" for i in range(16)]

# Read dataset using pandas
# header=None because dataset has no header row
# names=col_names assigns custom column names

data = pd.read_csv(url, header=None, names=col_names)


# ----------------------------------------------------------
# STEP 3 : Separate Input and Output Data
# ----------------------------------------------------------

# X contains input features
# axis=1 means remove column vertically

X = data.drop("letter", axis=1).values

# y contains target labels (A-Z letters)

y = data["letter"].values


# ----------------------------------------------------------
# STEP 4 : Label Encoding using LabelBinarizer
# ----------------------------------------------------------

# Create LabelBinarizer object
# Used for one-hot encoding output labels

encoder = LabelBinarizer()

# Convert letters into binary vectors

# Example:
# A -> [1 0 0 0 ...]
# B -> [0 1 0 0 ...]

Y = encoder.fit_transform(y)

# Why needed?
# Neural networks cannot understand text labels directly


# ----------------------------------------------------------
# STEP 5 : Split Dataset into Training and Testing
# ----------------------------------------------------------

# test_size=0.2
# 20% data used for testing

# random_state=42
# ensures same random split every time

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# ----------------------------------------------------------
# STEP 6 : Feature Scaling
# ----------------------------------------------------------

# Create StandardScaler object
# Used to normalize feature values

scaler = StandardScaler()

# fit_transform():
# learns mean/std deviation and scales training data

X_train = scaler.fit_transform(X_train)

# transform():
# scales test data using SAME scaling parameters

# Important:
# Avoids data leakage

X_test = scaler.transform(X_test)


# ----------------------------------------------------------
# STEP 7 : Build Deep Neural Network Model
# ----------------------------------------------------------

# Sequential model:
# layers added one after another

model = Sequential([

    # First hidden layer
    # 128 neurons
    # ReLU activation introduces nonlinearity

    # input_shape=(16,)
    # because dataset contains 16 features

    Dense(128, activation='relu', input_shape=(16,)),


    # Second hidden layer
    # 64 neurons for deeper learning

    Dense(64, activation='relu'),


    # Output layer
    # 26 neurons because:
    # total 26 alphabet classes (A-Z)

    # softmax activation:
    # converts outputs into probabilities

    Dense(26, activation='softmax')
])


# ----------------------------------------------------------
# STEP 8 : Compile Deep Learning Model
# ----------------------------------------------------------

model.compile(

    # Adam optimizer updates weights efficiently
    optimizer=Adam(learning_rate=0.001),

    # categorical_crossentropy used for
    # multiclass classification

    loss="categorical_crossentropy",

    # accuracy used for performance evaluation
    metrics=["accuracy"]
)


# ----------------------------------------------------------
# STEP 9 : Train Neural Network
# ----------------------------------------------------------

# fit() trains the model

# epochs=25
# complete passes through dataset

# batch_size=32
# 32 samples processed together

# validation_split=0.1
# 10% training data used for validation

history = model.fit(
    X_train,
    Y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1
)


# ----------------------------------------------------------
# What Actually Happens During Training?
# ----------------------------------------------------------

# Neural network:

# 1. Takes feature values as input
# 2. Passes data through hidden layers
# 3. Learns patterns of alphabet letters
# 4. Predicts probabilities for A-Z classes
# 5. Calculates prediction error
# 6. Updates weights using Adam optimizer
# 7. Repeats for 25 epochs to improve accuracy


# ----------------------------------------------------------
# STEP 10 : Evaluate Model Performance
# ----------------------------------------------------------

# evaluate() checks performance on unseen test data

test_loss, test_acc = model.evaluate(X_test, Y_test)

# Display final test accuracy

print(f"Test Accuracy: {test_acc:.4f}")


# ----------------------------------------------------------
# Meaning of Final Output
# ----------------------------------------------------------

# test_loss:
# prediction error on testing data

# test_acc:
# percentage of correct predictions

# Higher accuracy means:
# better alphabet classification performance
#-----------------------------------------------------------

# overview of code :-
# Load dataset
# Separate input features and output labels
# Convert labels into one-hot encoded vectors
# Split dataset into training and testing data
# Normalize input features using StandardScaler
# Build Deep Neural Network model
# Compile model using optimizer and loss function
# Train model using training data
# Evaluate model on test data
# Predict alphabet classes using trained model