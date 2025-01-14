from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Modell laden
model = load_model('lstm_model_hlocv.keras')

# Testdaten laden
X_test = np.load('stock_test_X.npy')
y_test = np.load('stock_test_y.npy')

# y_test ist bereits ein NumPy-Array, also direkt verwenden
y_test_np = y_test

# Rückskalierung vorbereiten
scaler_y_stock_test = MinMaxScaler(feature_range=(0, 1))
scaler_y_stock_test.fit(y_test_np.reshape(-1, 1))  # Passe den Scaler auf die Zielwerte an

# Einzelnes Beispiel vorhersagen
example_index = 0
X_example = X_test[example_index].reshape(1, -1, X_test.shape[2])  # Form für das Modell anpassen
y_example = y_test_np[example_index]

# Vorhersage
predicted_price = model.predict(X_example)

# Rückskalierung
predicted_price_rescaled = scaler_y_stock_test.inverse_transform(predicted_price)

# actual_price sollte ebenfalls in einem 2D-Array sein (mit einer Zeile und einer Spalte)
actual_price_rescaled = scaler_y_stock_test.inverse_transform(np.array([[y_example]]))  # 2D-Array

# Ergebnis ausgeben
print(f"Actual Price: {actual_price_rescaled[0][0]:.2f}, Predicted Price: {predicted_price_rescaled[0][0]:.2f}")
