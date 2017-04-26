from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from collections import Counter
import pandas as pd
import numpy as np
import csv
import re
import sys

def read_csv(file):
    with open(file, "rb") as csvfile:
        f = csv.reader(csvfile, dialect='excel')
        data = []
        for row in f:
            data.append(row)
    return data

def tokenize(comments):
    """
    Tokenize a list of texts 
    Return the rates of FUNCTION_WORDS and PUNCTUATIONS
    """
    text = ""
    for line in comments:
    # if a tweet has multiple lines
        text += line + " "
    text = text.lower()
    # tokenize a string; seperate abbreviations like I'm
    words_symbols = re.compile(r"[.,!\/#$%\^&\*;:{}=\-_`~()]|'\w+|\w+") 
    words = re.findall(words_symbols, text)
    return words

def find_most_common_words(data, N):
    """
    Find the most N common words
    """
    word_dic = Counter()
    for row in data:
        words = tokenize(row[1:])
        for word in words:
            word_dic[word] += 1
    frequent = word_dic.most_common(N)
    common_words = [key for key, value in frequent]
    return common_words

def get_X_Y(data, common_words):
    """
    Transform a list of texts into features and responses

    postive: 1
    neutral: 0
    negative: -1
    """
    Y = [1 if row[0]=='positive' else \
        (0 if row[0]=='neutral' else -1) for row in data]
    X = [get_ratio(row[1:], common_words) for row in data]
    return X, Y

def get_ratio(comments, common_words):
    """
    Determine the word rates
    """
    words = tokenize(comments)
    counts = Counter(words)
    ratios = []
    for word in common_words:
        ratios.append(counts[word] / float(len(words)) )
    if(sum(ratios) > 1.0001):
        print "Need to debug tokenize function"
        exit()
    return ratios

"""
Main Program
"""

# python my_script.py train.csv test.csv

# import warnings
# warnings.filterwarnings("ignore")

train = read_csv(sys.argv[1])
test = read_csv(sys.argv[2])

# Extract features and responses
commen_words = find_most_common_words(train, 400)
x_train, y_train = get_X_Y(train, commen_words)
x_test, y_test = get_X_Y(test, commen_words)

"""
Estiamte misclassification rates using LR, LDA, QDA, KNN
"""

# Logistic regression
try:
    C = [1.0, 10.0, 50.0, 100.0]
    param_grid = {"C":C}
    grid_search = GridSearchCV(linear_model.LogisticRegression(), param_grid, cv=3)
    grid_search.fit(x_train, y_train)
    y_pred = grid_search.predict(x_test)
    correct = np.array(y_test) == np.array(y_pred)
    error = 1.0 - sum(correct) / float(len(y_test))
    print "For logit, Misclassification rate = {}".format(error)
except RuntimeError:
    print "Runtime Error for the logistic regression"

# Linear Discriminant Analysis
try:
    LDA = LinearDiscriminantAnalysis(shrinkage=None)
    LDA.fit(x_train, y_train)
    y_pred = LDA.predict(x_test)
    correct = np.array(y_test) == np.array(y_pred)
    error = 1.0 - sum(correct) / float(len(y_test))
    print "For LDA, misclassification rate = {}".format(error)
except RuntimeError:
    print "Runtime Error for LDA"

# Quadratic Discriminant Analysis
# try:
#     QDA = QuadraticDiscriminantAnalysis()
#     QDA.fit(x_train, y_train)
#     y_pred = QDA.predict(x_test)
#     correct = np.array(y_test) == np.array(y_pred)
#     error = 1.0 - sum(correct) / float(len(y_test))
#     print "For QDA, misclassification rate = {}}".format(error)
# except RuntimeError:
#     errors.append(1.0)
#     print "Runtime Error for QDA"

# KNN
try:
    n_neighbors = [3, 10, 20, 30]
    param_grid = {"n_neighbors":n_neighbors}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 3)
    grid_search.fit(x_train, y_train)
    n = grid_search.best_params_['n_neighbors']
    y_pred = grid_search.predict(x_test)
    correct = np.array(y_test) == y_pred
    error = 1.0 - sum(correct) / float(len(y_test))
    print "For KNN, misclassification rate = {} when k = {}".\
            format(error, n)
except RuntimeError:
    print "Runtime Error for KNN"