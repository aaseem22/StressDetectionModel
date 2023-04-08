import numpy as np


class ComplementNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.class_map = {}
        labels = np.unique(y)
        for i, label in enumerate(labels):
            self.class_map[label] = i
        y = np.array([self.class_map[label] for label in y])
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        self.class_counts = np.zeros(num_classes, dtype=int)
        self.feature_counts = np.zeros((num_classes, num_features), dtype=float)
        self.feature_totals = np.zeros(num_features, dtype=float)
        self.class_totals = np.zeros(num_classes, dtype=float)

        # Count occurrences of each class and each feature for each class
        for i in range(num_samples):
            class_idx = np.where(self.classes == y[i])[0][0]
            self.class_counts[class_idx] += 1
            self.feature_counts[class_idx] += X[i]
            self.class_totals[class_idx] += np.sum(X[i])
            self.feature_totals += X[i]

        # Calculate complement probabilities for each class and feature
        self.complement_probs = np.zeros((num_classes, num_features), dtype=float)
        for i in range(num_classes):
            complement_counts = self.feature_totals - self.feature_counts[i]
            complement_totals = np.sum(self.feature_totals) - self.class_totals[i]
            self.complement_probs[i] = (complement_counts + self.alpha) / (
                        complement_totals + self.alpha * num_features)

        # Calculate prior probabilities for each class
        self.prior_probs = self.class_counts / np.sum(self.class_counts)


    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            class_probs = np.log(self.prior_probs) + np.sum(np.log(self.complement_probs[:, X[i] == 1]), axis=1)
            predicted_class = np.argmax(class_probs)
            predictions[i] = predicted_class
        return np.array(["Stress" if p == 1 else "No Stress" for p in predictions])

