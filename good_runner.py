import csv
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random

def load_data(sheet):
    csvfile = open(sheet, encoding = "ISO-8859-1",mode='r+')
    lines = csv.reader(csvfile)
    data = list(lines)
    data = data[1:]
    ham = [point for point in data if point[0] == 'ham']
    spam = [point for point in data if point[0] == 'spam']
    data = spam + random.sample(ham, len(spam))
    x = np.array([point[1] for point in data])
    print(x[0])
    y = np.array([point[0] for point in data])
    print(y[0])
    X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

train_data, test_data, train_labels, test_labels = load_data('spam.csv')
vectorizer = TfidfVectorizer()
vectorised_train_data = vectorizer.fit_transform(train_data)

classifier = LinearSVC()
classifier.fit(vectorised_train_data, train_labels)

print("Training complete!")

user_input = input("Send me a message!\n")
user_data = np.array([user_input])
vectorised_user_data = vectorizer.transform(user_data)

while(user_input != "quit"):
	prediction = classifier.predict(vectorised_user_data)
	print(prediction)
	user_input = input("Send me a message!\n")
	user_data = np.array([user_input])
	vectorised_user_data = vectorizer.transform(user_data)
print("Bye!")

