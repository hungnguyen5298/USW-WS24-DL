import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


X_train = np.load('../model/HLOCV/stock_train_X.npy', allow_pickle=True)
print(X_train)
# Korrelationsmatrix


corr_matrix = np.corrcoef(X_train.reshape(-1, X_train.shape[2]), rowvar=False)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
