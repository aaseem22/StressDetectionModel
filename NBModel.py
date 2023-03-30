import math

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


