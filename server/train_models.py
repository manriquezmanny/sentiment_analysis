# Imports
import keras.regularizers
import keras.utils
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# Checking for GPU
tf.config.list_physical_devices('GPU')

# Loading vectorized tweets and labels
data = np.load("./vectorized_data/vectorized_tweets1.2.npz")
vectorized_tweets = data["tweets"]
labels = data["labels"]

# Splitting the data into a training set and a validation set. Note, the test_size of 0.2 means 20% will be used for testing and 80% for training. Also random_state acts as a seed of how the data is split.
X_train, X_val, y_train, y_val = train_test_split(vectorized_tweets, labels, test_size=0.2, random_state=12)


# Defining my sequential model.
# Note, the embedding layer takes 50,000 as input representing 50,000 tokens/words most commonly used.
model = Sequential([
    Embedding(input_dim=50000, output_dim=128, input_length=100),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(pool_size=3),
    Conv1D(64, 3, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),  # Dense with L2 regularization
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])


# Defining Early Stopping conditions to avoid overfitting and stop training on convergence.
early_stopping = EarlyStopping(monitor="val_accuracy", patience=2)

# Compiling the model with a specified optimizer that performs stochastic gradient descent. Chose binary_crossentropy as loss function.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the model using the fit method. Here I specify the data, and the number of epochs.
# With these parameters, the model will stop training when the val_accuracy does not notably improve in 2 epochs.
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Using the evaluate method to evaluate the accuracy of the model.
test_loss, test_acc = model.evaluate(X_val, y_val)

# Saving the model so I can use it later. Saving as .keras format, a typical format to save models.
model.save("./models/model1.2.keras")