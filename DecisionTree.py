import numpy as np
from matplotlib._mathtext import Node
from nltk import FreqDist
from nltk.probability import entropy


class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, x_train, y_train):
        self.tree = self.build_tree(x_train, y_train)

    def predict(self, x_test):
        predictions = [self.traverse_tree(x, self.tree) for x in x_test]
        return np.array(predictions)

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] < node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def build_tree(self, X, y):
        num_samples, num_features = X.shape
        if num_samples == 0:
            return Node(value=None)
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        best_feature, best_threshold = self.get_best_split(X, y, num_samples, num_features)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left = self.build_tree(X[left_indices], y[left_indices])
        right = self.build_tree(X[right_indices], y[right_indices])
        return Node(best_feature, best_threshold, left, right)

    def get_best_split(self, X, y, num_samples, num_features):
        best_gain = -1
        split_index, split_threshold = None, None
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                gain = self.get_information_gain(y, feature_values, threshold, num_samples)
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
        return split_index, split_threshold

    def get_information_gain(self, y, feature_values, split_threshold, num_samples):
        y_freqdist = FreqDist(y)
        parent_entropy = entropy(y_freqdist)
        left_indices = feature_values < split_threshold
        right_indices = feature_values > split_threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        left_freqdist = FreqDist(y[left_indices])
        right_freqdist = FreqDist(y[right_indices])
        child_entropy = ((len(y[left_indices]) / num_samples) * entropy(left_freqdist.freqdist())
                         + (len(y[right_indices]) / num_samples) * entropy(right_freqdist.freqdist()))
        ig = parent_entropy - child_entropy
        return ig

