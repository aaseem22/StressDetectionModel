import numpy as np

# function to calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# function to calculate the variance of a list of numbers
def variance(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers) - 1)
    return variance

# function to calculate the covariance between two lists of numbers
def covariance(x, mean_x, y, mean_y):
    covariance = 0.0
    for i in range(len(x)):
        covariance += (x[i] - mean_x) * (y[i] - mean_y)
    return covariance / float(len(x) - 1)

# function to calculate the coefficients of linear regression
def coefficients(train_x, train_y):
    mean_x, mean_y = mean(train_x), mean(train_y)
    b1 = covariance(train_x, mean_x, train_y, mean_y) / variance(train_x)
    b0 = mean_y - b1 * mean_x
    return [b0, b1]


# function to perform linear regression
def linear_regression(train_x, train_y, test_x):
    b0, b1 = coefficients(train_x, train_y)
    predicted_y = b0 + b1 * test_x
    return predicted_y


# example usage
train_x = np.array([1, 2, 3, 4, 5])
train_y = np.array([2, 3, 4, 5, 6])
test_x = np.array([6])
predicted_y = linear_regression(train_x, train_y, test_x)
print('Input x: {}, Predicted y: {}'.format(test_x[0], predicted_y[0]))
