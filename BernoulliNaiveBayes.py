import numpy as np

class BernoulliNaiveBayes18:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        # Compute priors
        self.class_counts = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            self.class_counts[i] = np.sum(y == c)
        self.priors = self.class_counts / len(y)

        # Compute likelihoods
        self.feature_likelihoods = np.zeros((self.n_classes, self.n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.feature_likelihoods[i, :] = (np.sum(X_c, axis=0) + self.alpha) / (self.class_counts[i] + 2 * self.alpha)

    def predict(self, X):
        y_pred = np.empty(X.shape[0], dtype='object')
        for i, x in enumerate(X):
            posterior_probs = np.zeros(self.n_classes)
            for j in range(self.n_classes):
                likelihood = self.feature_likelihoods[j, :]
                likelihood_complement = 1 - likelihood
                log_likelihood = np.sum(np.log(np.where(x == 1, likelihood, likelihood_complement)))
                posterior_probs[j] = np.log(self.priors[j]) + log_likelihood
            y_pred[i] = self.classes[np.argmax(posterior_probs)]
        return y_pred

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        correct = np.sum(y_pred == y)
        total = len(y)
        return correct / total

    def precision(self, X, y, positive_class):
        y_pred = self.predict(X)
        true_positives = np.sum((y_pred == positive_class) & (y == positive_class))
        false_positives = np.sum((y_pred == positive_class) & (y != positive_class))
        bernauligph = true_positives / (true_positives + false_positives)
        return bernauligph

