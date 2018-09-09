
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(sheet):
    csvfile = open(sheet, encoding = "ISO-8859-1",mode='r+') # loads the data sheet into Python
    # sets encoding so it can read all characters
    # sets mode to r+ so it can read the file and modify it if necessary
    lines = csv.reader(csvfile) # command to read the file
    data = list(lines) # turns the file into a list that our algorithm can handle
    data = data[1:] # chops off the header row
    x = np.array([point[1] for point in data]) # creates an array of the text data points
    y = np.array([point[0] for point in data]) # creates an array of the labels corresponding to data points
    # needs to be an np array because the train_test_split function requires it
    X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2) 
    # randomly splits the data into train and test
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST # returns training data & labels and testing data & labels

train_data, test_data, train_labels, test_labels = load_data('spam.csv') # calls load_data on our data

vectorizer = TfidfVectorizer() # creates an instance of a TFIDF vectorizer
vectorised_train_data = vectorizer.fit_transform(train_data) # calls the vectorizer to transform training data
# fit_transform is used instead of transform because we need to fit the training data corpus to the algorithm
# later, when we vectorize testing data, we call transform 
    # because we have fixed our vectorization system based on the training data

classifier = LinearSVC() # instance of the SVM classifier with a linear kernel
classifier.fit(vectorised_train_data, train_labels) # fits training data to the classifier algorithm

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

