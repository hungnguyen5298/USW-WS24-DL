import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Prepare data for model
X_train = np.load('stock_train_X.npy')
y_train = np.load('stock_train_y.npy')
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Dataset in Batches
batch_size = 2  # Empfohlene Batchgröße
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.batch(batch_size, drop_remainder=True)

# Modell definieren
input_size = X_train.shape[2]
seq_len = X_train.shape[1]

model = Sequential([
    LSTM(64, input_shape=(seq_len, input_size), return_sequences=False),
    Dense(1, activation = 'sigmoid'),  # Keine Aktivierung für Regression
])

model.summary()

# Modell kompilieren
model.compile(
    loss='binary_crossentropy',  # Verlustfunktion für Regression
    optimizer='adam',
    metrics=['accuracy', 'AUC']  # Mean Absolute Error als Metrik
)

# Modell trainieren
history = model.fit(dataset, epochs=200)

# Modell evaluieren
loss, accuracy, auc = model.evaluate(dataset)
print(f"Loss: {loss}, Accuracy: {accuracy}, AUC: {auc}")

# Modell speichern
model.save('lstm_model_hlocv.keras')

# Verlustfunktion
plt.plot(history.history['loss'], label='Train Loss')
plt.legend()
plt.show()

# Genauigkeit
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.legend()
plt.show()

# AUC
plt.plot(history.history['AUC'], label='Train AUC')
plt.legend()
plt.show()