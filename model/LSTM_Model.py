import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Define Model

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dense(units=64, activation='relu'),
])

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
X_train = ... # input data with shape (batch_size, 10, 1)
y_train = ... # target data with shape (batch_size, 1)
model.fit(X_train, y_train, epochs=10)