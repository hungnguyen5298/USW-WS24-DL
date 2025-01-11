import tensorflow as tf
import numpy as np

# Beispiel: Dummy-Daten im Format (1431, 3, 10)
X_train_stock = np.random.rand(1431, 3, 10)  # Zufallsdaten als Beispiel

# Umwandlung in TensorFlow-Tensor
X_train_stock = tf.convert_to_tensor(X_train_stock, dtype=tf.float32)

print(X_train_stock)  # (1431, 3, 10)
