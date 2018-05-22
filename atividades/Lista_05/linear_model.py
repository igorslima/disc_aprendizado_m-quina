import numpy as np


class LogisticRegression:

    def fit(x,y, epochs=10, learning_rate=0.0001):
        beta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        for step in np.arange(epochs):
            x_beta = np.dot(x, beta)
            y_hat = 1/(1+ np.exp(-x_beta))
            likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
            preds = np.round(y_hat)
            accuracy = np.sum(preds == y)/len(preds)
            gradient = np.dot(np.transpose(x), y - y_hat)
            beta = beta + learning_rate * gradient
            if(step % 10 == 0):
                print("Depois de {} steps, semelhança {}, acurácia: {}".format(step,likelihood, accuracy))
        self.beta = beta
    def predict(self, x):
        b0 = self.beta[0]
        b1 = self.beta[1]
        b2 = self.beta[2]
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        return np.round(1.0 / (1 + np.exp(-(b0 + b1 * x1 + b2 * x2))))