import pandas as pd
import numpy as np
def euclidiana(X, x_row):
    X_ = (X - x_row) ** 2
    return np.sum(X_, axis=1) ** 0.5

def minkowski_distance(X, row, p):
    X_ = np.abs(X-row) ** p
    return np.sum(X_, axis=1) ** 1/p

def manhattan_distance(X, row):
    return minkowski_distance(X, row, 1)

def chebyshev_distance(X, row):
    return np.max(np.abs(X - row))
