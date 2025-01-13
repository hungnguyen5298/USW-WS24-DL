import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Daten laden, vorbereiten und konvertieren
X_train = np.load('stock2_train_X.npy')
y_train = np.load('stock2_train_y.npy')
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_val = np.load('stock2_val_X.npy')
y_val = np.load('stock2_val_y.npy')
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

X_test = np.load('stock2_test_X.npy')
y_test = np.load('stock2_test_y.npy')
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Batch-Größe
batch_size = 32

# TensorFlow-Datasets erstellen
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size, drop_remainder=True)

# Modell definieren
input_size = X_train.shape[2]
seq_len = X_train.shape[1]

model = Sequential([
    LSTM(32, input_shape=(seq_len, input_size), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.summary()

# Modell kompilieren
model.compile(
    loss='binary_crossentropy',
    #optimizer=SGD(learning_rate=0.0001),
    optimizer= Adam(learning_rate=0.005),
    metrics=['accuracy']
)

# Early Stopping Callback hinzufügen
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Modell trainieren
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping]
)

# Modell evaluieren
test_results = model.evaluate(test_dataset, return_dict=True)
print("\nTest Results:")
for metric, value in test_results.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Modell speichern
model.save('lstm_model_hlocv_test2.keras')

# Vorhersagen (Klassenwahrscheinlichkeiten)
predictions = model.predict(X_test)

# Vorhersagen in Klassen umwandeln (falls Wahrscheinlichkeiten vorliegen)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Ergebnisse speichern und visualisieren
plt.figure(figsize=(10, 6))
plt.plot(y_test.numpy(), label='Echte Klassen')
plt.plot(predicted_classes, label='Vorhergesagte Klassen', alpha=0.7)
plt.legend()
plt.title('Vorhersagen vs. Echte Klassen')
plt.show()

# Klassifikationsmetriken (optional)
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test.numpy(), predicted_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test.numpy(), predicted_classes))
