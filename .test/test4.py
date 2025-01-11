import matplotlib.pyplot as plt
import numpy as np

y_train = np.load('../model/HLOCV/stock_train_y.npy')
y_test = np.load('../model/HLOCV/stock_test_y.npy')
y_val = np.load('../model/HLOCV/stock_val_y.npy')

unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))


'''plt.hist(y_train, bins=50, alpha=0.5, label='Train Data')
plt.hist(y_val, bins=50, alpha=0.5, label='Validation Data')
plt.hist(y_test, bins=50, alpha=0.5, label='Test Data')
plt.legend()
plt.show()'''
