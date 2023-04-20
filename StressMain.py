import pandas as pd
import numpy as np
import nltk
import re
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.naive_bayes import BernoulliNB

from BernoulliNaiveBayes import BernoulliNaiveBayes18
from ComplementNaiveBayes import ComplementNaiveBayes1
from LogisticRegression import Logistic1

data = pd.read_csv("Stress.csv")
print(data.head())
print(data.isnull().sum())

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")

stopword = set(stopwords.words('english'))


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


data["text"] = data["text"].apply(clean)

#
# text = " ".join(i for i in data.text)
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(stopwords=stopwords,
#                       background_color="white").generate(text)
# plt.figure( figsize=(15,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
print(data.head())

x = np.array(data["text"])
y = np.array(data["label"])

x, y = shuffle(x, y, random_state=42)

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.33,
                                                random_state=42)


def calculate_accuracy(model, xtest, ytest):
    output1 = model.predict(xtest)
    acc = accuracy_score(ytest, output1)
    return acc




#Logistic Regression
def LogisticAcc(txt):
    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    data1 = cv.transform([txt])
    output = model.predict(data1)
    accuracylr = model.predict(xtest)
    lr_acc = accuracy_score(ytest, accuracylr)
    return lr_acc

def Logistic(txt):
    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    data1 = cv.transform([txt])
    output = model.predict(data1)
    return output[0]

#Decision Tree
def DecisionTree(txt):
    modeldt = DecisionTreeClassifier()
    modeldt.fit(xtrain, ytrain)
    data1 = cv.transform([txt])
    output = modeldt.predict(data1)
    return output[0]

def DecisionTreeAcc(txt):
    modeldt = DecisionTreeClassifier()
    modeldt.fit(xtrain, ytrain)

    data1 = cv.transform([txt])
    accuracydt = modeldt.predict(xtest)
    dt_acc = accuracy_score(ytest, accuracydt)
    return dt_acc

def ComplimentNaiveBayes11(txt):
    modelcnb = BernoulliNB()
    modelcnb.fit(xtrain, ytrain)
    data1 = cv.transform([txt])
    output = modelcnb.predict(data1.toarray())
    return output[0]
def CNBAccuracy(txt):
    modelcnb = BernoulliNB()
    modelcnb.fit(xtrain, ytrain)
    data1 = cv.transform([txt])
    accuracycnb = modelcnb.predict(xtest.toarray())
    cnb_acc = accuracy_score(ytest, accuracycnb)
    return cnb_acc



