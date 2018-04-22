import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    def fit(self, X, y):
        self.b1 = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X)) ** 2)
        self.b0 = np.mean(y) - self.b1 * np.mean(X)
        
    def predict(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(self.b1 * X[i] + self.b0)
        return np.array(pred)