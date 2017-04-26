from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def KNN_MSE(train_x, train_y, test_x, test_y):
    # Estmate prediciton MSE using KNN
    n_neighbors = range(1, 4)
    param_grid = {"n_neighbors" : n_neighbors}
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv = 5)
    grid_search.fit(train_x, train_y)
    print grid_search.best_params_['n_neighbors']
    hypotheses = grid_search.predict(test_x)
    MSE = np.mean((test_y - hypotheses) ** 2.0)
    print "The MSE of KNN = {}".format(MSE)

def LR_MSE(train_x, train_y, test_x, test_y):
    # Estmate prediciton MSE using linear regression
    algo = linear_model.LinearRegression()
    algo.fit(train_x, train_y)
    hypotheses = algo.predict (test_x)
    MSE = np.mean((test_y - hypotheses) ** 2.0)
    print "The MSE of LR = {}".format(MSE)

"""
Boston data
"""
df = pd.read_csv("boston.csv")
train = df.iloc[1:-50, ]
test = df.iloc[-50:, ]

""" KNN """
KNN_MSE(train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1])
LR_MSE(train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1])




