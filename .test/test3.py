import numpy as np

X_train = np.load('../model/HLOCV/stock_test_X.npy')
y_train = np.load('../model/HLOCV/stock_test_y.npy')


print("X_train enthält NaN:", np.isnan(X_train).any())
print("y_train enthält NaN:", np.isnan(y_train).any())
print("X_train Max:", np.max(X_train), "Min:", np.min(X_train))
print("y_train Max:", np.max(y_train), "Min:", np.min(y_train))
