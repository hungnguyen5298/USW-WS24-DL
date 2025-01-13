import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

# Daten laden
stock2 = pd.read_csv('../../data_preprocessing/stock_data_feature_engineering/stock_data_apple_indicators.csv')

# Sortieren und unnötige Spalten entfernen
stock2 = stock2.sort_values(by='Timestamp', ascending=True)
stock2 = stock2.drop(columns=['Timestamp', 'Profit_Trend_Label'])

# Train-/Validierungs-/Test-Splits
split_index_stock2_train = int(len(stock2) * 0.7)
split_index_stock2_val = split_index_stock2_train + int(len(stock2) * 0.2)

stock2_X = stock2.drop(columns=['Close'])
stock2_y = stock2['Close']

stock2_X_train, stock2_X_val, stock2_X_test = (
    stock2_X[:split_index_stock2_train],
    stock2_X[split_index_stock2_train:split_index_stock2_val],
    stock2_X[split_index_stock2_val:]
)

stock2_y_train, stock2_y_val, stock2_y_test = (
    stock2_y[:split_index_stock2_train],
    stock2_y[split_index_stock2_train:split_index_stock2_val],
    stock2_y[split_index_stock2_val:]
)

# Feature-Skalierung (X)
scaler_X_train = MinMaxScaler(feature_range=(0, 1))
scaler_X_val = MinMaxScaler(feature_range=(0, 1))
scaler_X_test = MinMaxScaler(feature_range=(0, 1))

numerical_columns_stock2 = stock2_X.select_dtypes(include=['float64', 'int64']).columns

stock2_X_train[numerical_columns_stock2] = scaler_X_train.fit_transform(stock2_X_train[numerical_columns_stock2])
stock2_X_val[numerical_columns_stock2] = scaler_X_val.fit_transform(stock2_X_val[numerical_columns_stock2])
stock2_X_test[numerical_columns_stock2] = scaler_X_test.fit_transform(stock2_X_test[numerical_columns_stock2])

# Zielvariable skalieren (y)
scaler_y_train = MinMaxScaler(feature_range=(0, 1))
scaler_y_val = MinMaxScaler(feature_range=(0, 1))
scaler_y_test = MinMaxScaler(feature_range=(0, 1))

stock2_y_train = scaler_y_train.fit_transform(stock2_y_train.values.reshape(-1, 1))
stock2_y_val = scaler_y_val.fit_transform(stock2_y_val.values.reshape(-1, 1))
stock2_y_test = scaler_y_test.fit_transform(stock2_y_test.values.reshape(-1, 1))


# Sequenzen erstellen
def create_sequences(X, y, window_size):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - window_size):
        sequences_X.append(X[i:i + window_size])
        sequences_y.append(y[i + window_size])
    return np.array(sequences_X), np.array(sequences_y)

window_size = 20
X_train_stock2, y_train_stock2 = create_sequences(stock2_X_train.values, stock2_y_train, window_size)
X_val_stock2, y_val_stock2 = create_sequences(stock2_X_val.values, stock2_y_val, window_size)
X_test_stock2, y_test_stock2 = create_sequences(stock2_X_test.values, stock2_y_test, window_size)

# Daten vorbereiten
X_train, y_train = X_train_stock2, y_train_stock2
X_val, y_val = X_val_stock2, y_val_stock2
X_test, y_test = X_test_stock2, y_test_stock2

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
    loss='mean_squared_error',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['mae']
)

# Early Stopping
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
model.save('lstm_model_hlocv_test.keras')

# Vorhersagen und Rückskalierung
predictions = model.predict(X_test)
predictions_rescaled = scaler_y_test.inverse_transform(predictions)
y_test_rescaled = scaler_y_test.inverse_transform(y_test)

# Fehler berechnen
mae_rescaled = mean_absolute_error(y_test_rescaled, predictions_rescaled)
print(f"MAE (nach Rückskalierung): {mae_rescaled:.4f}")

# Plot Training/Validation Verlust
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test-Ergebnisse plotten
test_metrics = ['loss', 'mae']
test_values = [test_results[metric] for metric in test_metrics]

plt.bar(test_metrics, test_values)
plt.title("Test Metrics")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Echte Werte')
plt.plot(predictions_rescaled, label='Vorhersagen')
plt.legend()
plt.title('Vorhersagen vs. Echte Werte')
plt.show()


print(f"Skalierte Wertebereiche:")
print(f"Train (X): Min: {stock2_X_train.min().min()}, Max: {stock2_X_train.max().max()}")
print(f"Val (X): Min: {stock2_X_val.min().min()}, Max: {stock2_X_val.max().max()}")
print(f"Test (X): Min: {stock2_X_test.min().min()}, Max: {stock2_X_test.max().max()}")
print(f"Train (y): Min: {stock2_y_train.min()}, Max: {stock2_y_train.max()}")
print(f"Val (y): Min: {stock2_y_val.min()}, Max: {stock2_y_val.max()}")
print(f"Test (y): Min: {stock2_y_test.min()}, Max: {stock2_y_test.max()}")
