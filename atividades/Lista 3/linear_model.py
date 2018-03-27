import numpy as np
class SimpleLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    def step_gradiente(self,b0, b1, x, y, learning_rate):
        N = float(len(y))
        b0_gradient = 2/N * np.sum(-(y - ((b1 * x) + b0)))
        b1_gradient = 2/N * np.sum(-x * (y - ((b1 * x) + b0)))
        new_b0 = b0 - (learning_rate * b0_gradient)
        new_b1 = b1 - (learning_rate * b1_gradient)
        return new_b0, new_b1

    def run_gradiente(self,x, y, b0, b1, learning_rate, epochs):
        for _ in range(epochs):
            _b0, _b1 = self.step_gradiente(b0, b1, x,y, learning_rate)
        return _b0, _b1

    def run(self,x, y, initial_b0, initial_b1, learning_rate, epochs):
        self.b0, self.b1 = self.run_gradiente(x, y, initial_b0, initial_b1, learning_rate, epochs)

    def fit(self, x, y, learning_rate=0.0001, epochs=100000):
        self.run(x,y,0,0,learning_rate,epochs)

    def predict(self, x):
        return float(self.b1) * float(x) + float(self.b0)
