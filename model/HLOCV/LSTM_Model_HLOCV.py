import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2, l1


# Daten laden, vorbreiten und konvertieren
X_train = np.load('stock_train_X.npy')
y_train = np.load('stock_train_y.npy')
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_val = np.load('stock_val_X.npy')
y_val = np.load('stock_val_y.npy')
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

X_test = np.load('stock_test_X.npy')
y_test = np.load('stock_test_y.npy')
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

'''model = Sequential([
    LSTM(32, input_shape=(seq_len, input_size),return_sequences=False),
    Dropout(0.1),
    Dense(1, activation = 'sigmoid')
])'''

model = Sequential([
    LSTM(32, input_shape=(seq_len, input_size), return_sequences=True),
    LSTM(16, return_sequences=False),
    Dense(1, activation='sigmoid')
])

'''model = Sequential([
    LSTM(32, input_shape=(seq_len, input_size), return_sequences=False),
    #Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])'''

'''model = Sequential([
    GRU(128, input_shape=(seq_len, input_size), return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    GRU(64, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    GRU(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])'''

model.summary()

# Modell kompilieren
model.compile(
    loss='binary_crossentropy',
    #optimizer=SGD(learning_rate=0.0001),
    optimizer= Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# Early Stopping Callback hinzufügen
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# Modell trainieren
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=300,
    verbose=1,
    callbacks=[early_stopping]
)

# Modell evaluieren auf Testdaten und Tracking
test_results = model.evaluate(test_dataset, return_dict=True)
print("\nTest Results:")
for metric, value in test_results.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Modell speichern
model.save('lstm_model_hlocv.keras')

# Ergebnisse speichern und visualisieren
def plot_metric(history, metric, title):
    plt.plot(history.history[metric], label=f'Train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

# Verlust, Genauigkeit und AUC plotten
plot_metric(history, 'loss', 'Loss Over Epochs')
plot_metric(history, 'accuracy', 'Accuracy Over Epochs')

# Test-Ergebnisse visualisieren
test_metrics = ['loss', 'accuracy']
test_values = [test_results[metric] for metric in test_metrics]

plt.bar(test_metrics, test_values)
plt.title("Test Metrics")
plt.ylabel("Value")
plt.show()
