from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import ensemble
import pandas as pd
import numpy as np
import warnings
import sys

"""
Main Program
"""

# python my_a3_script.py input.csv output.text

if(len(sys.argv)) <3:
    print "please input python script name, input.csv, and output.text"
    sys.exit()
else:
    for i in range(1, len(sys.argv)):
        input_csv = sys.argv[1]
        output_name = sys.argv[2]

"""
Read data and get features and responses
"""

df_GDP = pd.read_csv("GDP by country and year.csv")
country_list = df_GDP.iloc[:, 0]
# Missing data filled with 0
df_GDP = df_GDP.fillna(0)
# One-hot encoding for variable "countries"
countries = pd.get_dummies(country_list)
countries.index = range(len(countries.index))
df_GDP = pd.concat([df_GDP, countries], axis = 1)
# melt GDP
primary_key_list = ['Country Name'] + country_list.tolist()
df_GDP = pd.melt(df_GDP, id_vars=primary_key_list, var_name='year', value_name='GDP')
df_GDP['year'] = df_GDP['year'].astype(int)
df_GDP['GDP'] = df_GDP['GDP'].astype(float)

df_life = pd.read_csv("life expectancy by country and year.csv")
# Missing data filled using linear regression
for j in range(len(df_life.index)):
# Fit a linear regression model for each row
    y = df_life.iloc[j, :][1:]
    isnull = pd.isnull(y)
    if np.sum(isnull.astype(int)) >= 1:
    # At least one nan
        x_test = y[isnull].index.values.astype(float)
        y_train = y[~isnull]
        x_train = y_train.index.values.astype(float)
        algo = linear_model.LinearRegression()
        algo.fit(x_train.reshape(-1,1), y_train)
        y_pred = algo.predict(x_test.reshape(-1, 1))
        # The first col excuded: y = df_life.iloc[j,:][1:]
        isnull = np.append([False], isnull)
        df_life.ix[j, isnull] = y_pred
# melt life
df_life = pd.melt(df_life, id_vars=['Country Name'], var_name='year', value_name='life')
df_life['year'] = df_life['year'].astype(int)
df_life['life'] = df_life['life'].astype(float)

"""
Get train and test data
"""

# Get features and responses
df = df_GDP.merge(df_life, how="inner", on=['Country Name', 'year'])
# GDP is completely wrong
x_train = df.iloc[:, 1:-2]
y_train = df.iloc[:, -1]

test = pd.read_csv(input_csv, header=None)
test_countries = test.iloc[:, 0].values
x_test = pd.DataFrame()
for country in test_countries:
    temp = (country_list == country)
    x_test = x_test.append(temp)
x_test.index = range(len(test_countries))
x_test['year'] = test.iloc[:, 1]
y_test = test.iloc[:, 2]

"""
Predict life expectancy with RandomForest
"""
# Random forest
n_estimators = [10, 50, 100, 150]
param_grid = {"n_estimators":n_estimators}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search.fit(x_train, y_train)
n = grid_search.best_params_['n_estimators']
y_pred = grid_search.predict(x_test)
MSE = np.mean((y_test - y_pred)**2.0)
print "When num of trees = {}, MSE = {}".format(n, MSE)

with open(output_name, 'w') as f:
    for s in y_pred:
        f.write(str(s) + '\n')