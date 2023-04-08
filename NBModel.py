import math

import numpy as np


# function to calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# function to calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# function to calculate the Gaussian probability density function
def gaussian_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# function to separate the data by class
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated

# function to calculate the mean, standard deviation and count of each attribute for each class
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute), len(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]  # remove the summary for the class
    return summaries


# function to train the Naive Bayes classifier
def train_naive_bayes(train_data):
    separated = separate_by_class(train_data)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

# function to predict the class for a new data instance
def predict_naive_bayes(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= gaussian_prob(x, mean, stdev)
    return max(probabilities, key=probabilities.get)


# example usage
train_data = [[1,2,0], [2,3,0], [3,3,0], [2,1,1], [3,2,1], [4,3,1]]
summaries = train_naive_bayes(train_data)
input_vector = [2,2]
class_prediction = predict_naive_bayes(summaries, input_vector)
print('Input vector: {}, Predicted class: {}'.format(input_vector, class_prediction))
#
#
# class BernoulliNB1:
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha  # Laplace smoothing parameter
#
#     def fit(self, X, y):
#         self.classes = np.unique(y)
#         self.num_classes = len(self.classes)
#         self.num_features = X.shape[1]
#         self.class_prior = np.zeros(self.num_classes)
#         self.feature_prob = np.zeros((self.num_classes, self.num_features))
#
#         for i, c in enumerate(self.classes):
#             X_c = X[y == c]
#             self.class_prior[i] = (len(X_c) + self.alpha) / (len(X) + self.alpha * self.num_classes)
#             self.feature_prob[i] = ((X_c.sum(axis=0) + self.alpha) / (len(X_c) + 2 * self.alpha))
#
#     def predict(self, X):
#         pred = []
#         for x in X:
#             pred_probs = []
#             for i in range(self.num_classes):
#                 prior_prob = np.log(self.class_prior[i])
#                 feature_probs = np.log(self.feature_prob[i]) * x + np.log(1 - self.feature_prob[i]) * (1 - x)
#                 pred_probs.append(np.sum(prior_prob + feature_probs))
#             pred.append(self.classes[np.argmax(pred_probs)])
#         return pred
#
#
# import numpy as np
#
#
# class NaiveBayesClassifier:
#
#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         self.classes = np.unique(y)
#         self.mean = np.zeros((len(self.classes), X.shape[1]))
#         self.var = np.zeros((len(self.classes), X.shape[1]))
#         self.prior = np.zeros(len(self.classes))
#
#         for i, c in enumerate(self.classes):
#             X_c = X[y == c]
#             self.mean[i, :] = X_c.mean(axis=0)
#             self.var[i, :] = X_c.var(axis=0)
#             self.prior[i] = X_c.shape[0] / X.shape[0]
#
#     def predict(self, X):
#         y_pred = np.zeros(X.shape[0])
#         for i, x in enumerate(X):
#             p = np.zeros(len(self.classes))
#             for j, c in enumerate(self.classes):
#                 p[j] = np.log(self.prior[j])
#                 p[j] += np.sum(np.log((1 / np.sqrt(2 * np.pi * self.var[j, :])) * np.exp(
#                     -(x - self.mean[j, :]) ** 2 / (2 * self.var[j, :]))))
#             y_pred[i] = self.classes[np.argmax(p)]
#         return y_pred
#
# import numpy as np
#
# class BernoulliNaiveBayes:
#
#     def __init__(self, alpha=1):
#         self.alpha = alpha
#
#     def fit(self, X, y):
#         self.classes = np.unique(y)
#         self.n_classes = len(self.classes)
#         self.n_features = X.shape[1]
#         self.priors = np.zeros(self.n_classes)
#         self.log_likelihoods = np.zeros((self.n_classes, self.n_features))
#
#         for i, c in enumerate(self.classes):
#             X_c = X[y == c]
#             self.priors[i] = X_c.shape[0] / X.shape[0]
#             self.log_likelihoods[i, :] = np.log((np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha))
#
#     def predict(self, X):
#         y_pred = np.zeros(X.shape[0])
#         for i, x in enumerate(X):
#             posterior_probs = np.zeros(self.n_classes)
#             for j in range(self.n_classes):
#                 posterior_probs[j] = np.log(self.priors[j]) + np.sum(x * self.log_likelihoods[j, :] + (1 - x) * np.log(1 - np.exp(self.log_likelihoods[j, :])))
#             y_pred[i] = self.classes[np.argmax(posterior_probs)]
#         return y_pred
#
#


import numpy as np
from sklearn.preprocessing import LabelEncoder

class BernoulliNaiveBayes4:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.log_likelihoods = np.zeros((self.n_classes, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / X.shape[0]
            self.log_likelihoods[i, :] = np.log((np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha))

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            posterior_probs = np.zeros(self.n_classes)
            for j in range(self.n_classes):
                posterior_probs[j] = np.log(self.priors[j]) + np.sum(x * self.log_likelihoods[j, :] + (1 - x) * np.log(1 - np.exp(self.log_likelihoods[j, :])))
            y_pred[i] = self.classes[np.argmax(posterior_probs)]
        return y_pred

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class BernoulliNaiveBayes5:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        # encode y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # encode X
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        X = onehot_encoder.fit_transform(X)

        # compute priors and log-likelihoods
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.log_likelihoods = np.zeros((self.n_classes, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / X.shape[0]
            self.log_likelihoods[i, :] = np.log((np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha))

    def predict(self, X):
        # encode X
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        X = onehot_encoder.fit_transform(X)

        # predict labels
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            posterior_probs = np.zeros(self.n_classes)
            for j in range(self.n_classes):
                posterior_probs[j] = np.log(self.priors[j]) + np.sum(x * self.log_likelihoods[j, :] + (1 - x) * np.log(1 - np.exp(self.log_likelihoods[j, :])))
            y_pred[i] = self.classes[np.argmax(posterior_probs)]
        return y_pred

