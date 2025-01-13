import numpy as np
import pandas as pd
# Load the data
X_train = np.load('../model/HLOCV/stock2_train_X.npy', allow_pickle=True)

input_size = X_train.shape[2]
seq_len = X_train.shape[1]

print(input_size)
print(seq_len)
print(X_train.shape)