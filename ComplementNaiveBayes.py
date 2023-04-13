
import numpy as np


class ComplementNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prior_probs = None
        self.complement_probs = None
        self.classes = None

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        self.prior_probs = np.zeros(num_classes)
        self.complement_probs = np.zeros((num_classes, num_features))

        for i, c in enumerate(self.classes):
            X_c = x[y == c]
            self.prior_probs[i] = X_c.shape[0] / num_samples
            self.complement_probs[i] = ((X_c.shape[0] - np.sum(X_c, axis=0)) + self.alpha) / (
                    x.shape[0] - np.sum(x, axis=0) + num_features * self.alpha)

    def predict(self, x):
        num_samples = x.shape[0]
        predictions = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            class_probs = np.log(self.prior_probs) + np.sum(np.log(self.complement_probs[:, x[i] == 1]), axis=1)
            predicted_class = np.argmax(class_probs)
            predictions[i] = predicted_class
        return np.array(["Stress 😔" if p == 1 else "No Stress😄" for p in predictions])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    precision = tp / (tp + fp)
    return precision



