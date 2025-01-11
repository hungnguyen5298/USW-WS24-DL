import numpy as np
import pandas as pd
# Load the data
X_train_stock = np.load('../model/HLOCV_VADER/vader_train_X.npy', allow_pickle=True)

print(X_train_stock.shape)
