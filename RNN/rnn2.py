import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.datasets import imdb

print("---------------------- Loading IMDb Dataset -------------------------\n")

# Load IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# Setting the maximum number of words to consider in the dataset
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("---------------------- Dataset Loaded -------------------------\n")
import numpy as np


# Pad sequences to a fixed length
max_length = 100
padded_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

print("---------------------- Data Preprocessed -------------------------\n")

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=24, input_length=max_length),
    tf.keras.layers.SimpleRNN(24, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("---------------------- Model Summary -------------------------\n")

# Summarize the model
print(model.summary())

print("---------------------- Training Model -------------------------\n")

# Fit the model
model.fit(x=padded_train, y=y_train, epochs=5, validation_data=(padded_test, y_test))

print("---------------------- Model Trained -------------------------\n")

# Evaluate the model
print("---------------------- Model Evaluation -------------------------\n")

preds = (model.predict(padded_test) > 0.5).astype("int32")

confusion_matrix(y_test, preds)

# Save the model
model.save("rnn_model.h5")

print("---------------------- Model Saved -------------------------\n")
