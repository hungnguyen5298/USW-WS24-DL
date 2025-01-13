import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2, l1

# Daten laden, vorbreiten und konvertieren
X_train = np.load('finbert_train_X.npy')
y_train = np.load('finbert_train_y.npy')
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_val = np.load('finbert_val_X.npy')
y_val = np.load('finbert_val_y.npy')
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

X_test = np.load('finbert_test_X.npy')
y_test = np.load('finbert_test_y.npy')
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Batch-Größe definieren
batch_size = 16

# Trainings- und Validierungsdatensatz erstellen
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
    loss='mean_squared_error',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['mae']
)

# Early Stopping Callback hinzufügen
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# Modell trainieren
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1#,
    #callbacks=[early_stopping]
)

# Modell evaluieren auf Testdaten und Tracking
test_results = model.evaluate(test_dataset, return_dict=True)
print("\nTest Results:")
for metric, value in test_results.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Modell speichern
model.save('lstm_model_hlocv_finbert.keras')

# Scaler anpassen (fit auf die Zielwerte, konvertiere y_train in ein NumPy-Array)
scaler_y_finbert_test = MinMaxScaler(feature_range=(0, 1))

# Konvertiere TensorFlow-Tensor in NumPy-Array
y_train_np = y_train.numpy()  # Konvertiere zu NumPy
scaler_y_finbert_test.fit(y_train_np.reshape(-1, 1))  # Passe den Scaler auf die Trainingsdaten an

# Vorhersagen
predictions = model.predict(X_test)

# Rückskalierung (konvertiere predictions in NumPy-Array, falls nötig)
predictions_np = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
predictions_rescaled = scaler_y_finbert_test.inverse_transform(predictions_np)

y_test_np = y_test.numpy()  # Konvertiere y_test in NumPy
y_test_rescaled = scaler_y_finbert_test.inverse_transform(y_test_np.reshape(-1, 1))

# Fehler berechnen
mae_rescaled = mean_absolute_error(y_test_rescaled, predictions_rescaled)
print(f"MAE (nach Rückskalierung): {mae_rescaled:.4f}")

'''# Test-Ergebnisse plotten
test_metrics = ['loss', 'mae']
test_values = [test_results[metric] for metric in test_metrics]

plt.bar(test_metrics, test_values)
plt.title("Test Metrics")
plt.ylabel("Value")
plt.show()'''

# Vorhersagen vs. echte Werte plotten
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Echt')
plt.plot(predictions_rescaled, label='Vorhersagen')
plt.legend()
plt.title('Vorhersagen vs. Echte Preis von AAPL mit HLOCV-Daten + Sentimentdaten aus FinBERT')
plt.show()

# Learning Curve: Verlust (Loss) plotten
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Learning Curve (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Learning Curve: MAE plotten (falls getrackt)
if 'mae' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE', color='green')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.title('Learning Curve (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.show()

residuals = y_test_rescaled - predictions_rescaled

# Histogramm der Residuen
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='purple', alpha=0.7)
plt.title('Histogramm der Residuen')
plt.xlabel('Fehler (Residuen)')
plt.ylabel('Häufigkeit')
plt.grid(True)
plt.show()

# Residuen über Zeit
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuen', color='brown')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residuen über Zeit')
plt.xlabel('Zeitpunkte')
plt.ylabel('Fehler')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test_rescaled, predictions_rescaled, alpha=0.6, color='darkblue')
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], color='red', linestyle='--', linewidth=2)
plt.title('Scatter-Plot: Vorhersagen vs. Echte Werte')
plt.xlabel('Echte Werte')
plt.ylabel('Vorhersagen')
plt.grid(True)
plt.show()