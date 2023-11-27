import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the IMDb dataset
max_features = 10000  # consider the top 10,000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Padding sequences to the same length
maxlen = 100  # you can adjust this based on your requirements
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Modelling a sample DNN
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(maxlen,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary classification, so use 'sigmoid'

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {round(accuracy * 100, 2)}%')



# Save the model
model.save("dnn_model.h6")

