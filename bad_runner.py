import csv
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(sheet):
    csvfile = open(sheet, encoding = "ISO-8859-1",mode='r+')
    lines = csv.reader(csvfile)
    data = list(lines)
    data = data[1:]
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

print("Training complete!") # to let user know that they can start typing

user_input = input("Send me a message!\n") # function to take user input
user_data = np.array([user_input]) # needs to be an array so vectorizer can process
vectorised_user_data = vectorizer.transform(user_data) # vectorizes the text according to training fit

while(user_input != "quit"): # runs this until you type in "quit"
	prediction = classifier.predict(vectorised_user_data) # calculates whether text is ham or spam
	print(prediction)
	user_input = input("Send me a message!\n") # asks for user input again!
	user_data = np.array([user_input]) # repeat the user input processing sequence
	vectorised_user_data = vectorizer.transform(user_data) # goes into next loop with vectorized data
print("Bye!")

