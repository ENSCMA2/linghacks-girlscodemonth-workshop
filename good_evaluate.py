import csv
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer

import random

def load_data(sheet):
    csvfile = open(sheet, encoding = "ISO-8859-1",mode='r+')
    lines = csv.reader(csvfile)
    data = list(lines)
    data = data[1:]
    # next 2 lines separate ham and spam data so we can sample separately from each
    ham = [point for point in data if point[0] == 'ham']
    spam = [point for point in data if point[0] == 'spam']
    data = spam + random.sample(ham, len(spam)) # randomly sample as many spams as hams
    x = np.array([point[1] for point in data])
    y = np.array([point[0] for point in data])
    X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

train_data, test_data, train_labels, test_labels = load_data('spam.csv')

print(test_labels)
vectorizer = TfidfVectorizer()
vectorised_train_data = vectorizer.fit_transform(train_data)

classifier = LinearSVC()
classifier.fit(vectorised_train_data, train_labels)

vectorised_test_data = vectorizer.transform(test_data)

predictions = classifier.predict(vectorised_test_data)

accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, pos_label = 'spam')
recall = recall_score(test_labels, predictions, pos_label = 'spam')
f1 = f1_score(test_labels, predictions, pos_label = 'spam')


print(str(accuracy))
print(str(precision))
print(str(recall))
print(str(f1))

