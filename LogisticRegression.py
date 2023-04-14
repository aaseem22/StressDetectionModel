import numpy as np
import scipy


class LogisticRegression11:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize parameters
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # gradient descent
        for i in range(self.num_iterations):
            # calculate predicted values
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(y_pred)

    def _sigmoid(self, z):
        if isinstance(z, scipy.sparse.csr_matrix):
            z = z.toarray()
        return 1 / (1 + np.exp(-z))




