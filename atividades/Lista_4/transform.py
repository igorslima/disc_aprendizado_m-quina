import numpy as np

def normalize(x):
    x_norm = np.copy(x)
    n_cols = x.shape[1]
    for i in range(n_cols):
        x_norm[:,i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))
    return x_norm

def standardize(X):
    X_std = np.copy(X)
    n_cols = X.shape[1]
    for i in range(n_cols):
        X_std[: , i] = (X[:, i] - np.min(X[:, i])) / np.std(X[:,i])
    return X_std