import sys
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import re

# Given raw data, estimate features
def get_features(df, mean_temp_dic):
    print "melt"
    df = pd.melt(df, id_vars=['station', 'month', 'day'], 
                        var_name='hour', value_name='temp')
    print "sort values"
    df = df.sort_values(by = ['station', 'month', 'day', 'hour'], \
                        ascending = [True, True, True, True])
    print "prev_temp"
    df['prev_temp'] = df.groupby(by = ['station'])['temp'].shift(1)
    print "prev_day_temp"
    df['prev_day_temp'] = df.groupby(by = ['station'])['temp'].shift(24)
    print "run_mean_temp"
    df['run_mean_temp'] = df.groupby(by = ['station', 'month', 'day'])['temp'].\
                            transform(lambda x: x.rolling(window=24, min_periods=1).mean().shift(1))
    df['datetime'] = df['month'] + df['day'] + df['hour']
    print "mean_temp"
    datetime = df['datetime'].tolist()
    df['mean_temp'] = [mean_temp_dic[val] for val in datetime]
    df = df.dropna()
    print df.head()
    return df[['prev_temp', 'prev_day_temp', 'run_mean_temp', 'mean_temp']], df['temp']

# Get testing stations
# test_stations = ["USW00023234", "USW00014918", "USW00012919", "USW00013743", "USW00025309"]
# command: python HW1_task2_KNN.py USW00023234 USW00014918 USW00012919 USW00013743 USW00025309

"""
Read test stations
"""
test_stations = []
if(len(sys.argv)) == 1: # if no station given
    print "no station given"
    sys.exit()
else:
    for i in range(1, len(sys.argv)):
        test_stations.append(sys.argv[i])

"""
Read txt files and clean data
Get df_train
"""

f = open("hly-temp-normal_full.txt", "r")
data = f.read()
f.close()
lines = data.strip().split('\n')
lines = [re.split(' +', line.strip()) for line in lines]
# Remove the character following the most temperature readings
lines = [[re.sub('[A-Z]', '', item) if re.search("[A-Z]$", item) else item \
        for item in line] \
        for line in lines]
# Create dataframe
hours = ['0'+str(i) if i <= 9 else str(i) for i in range(1, 25)]
df = pd.DataFrame(data = lines,
                  index = range(len(lines)),
                  columns=['station', 'month', 'day'] + hours)
for i in range(3, len(df.columns)):
    df.iloc[:, i] = df.iloc[:, i].astype(int)
# Imputation with col mean values
df = df.sort_values(by = ['station', 'month', 'day',])
df = df.replace(-9999, np.nan)
df = df.fillna(df.mean())

"""
Train test split
"""

df_train = df.ix[np.logical_not(np.in1d(df['station'], test_stations)), :]
df_test = df.ix[np.in1d(df['station'], test_stations), :]

"""
Feature 1:
Estimate the mean temperature of that hour's reading (across all stations) on that day
"""

df_train_data = pd.melt(df.ix[:, 1:], id_vars=['month', 'day'], 
                        var_name='hour', value_name='temp')
train_mean_temp = df_train_data.groupby(['month', 'day', 'hour'], \
                                        as_index=False)['temp'].aggregate(np.mean)
train_mean_temp['date'] = train_mean_temp['month'] + train_mean_temp['day'] + train_mean_temp['hour']
train_temp_dic = dict(zip(train_mean_temp['date'], train_mean_temp['temp']))

"""
Other features
"""

(train_x, train_y) = get_features(df_train, train_temp_dic)
(test_x, test_y) = get_features(df_test, train_temp_dic)

"""
Estiamte MSE using two approaches: SLR and KNN
"""

# SLR
algo = linear_model.LinearRegression()
algo.fit(train_x, train_y)
y_pred = algo.predict(test_x)
MSE_LR = np.mean((test_y - y_pred)**2.0)
print "The MSE of LR = {}".format(MSE_LR)

# KNN
n_neighbors = range(1, 52, 5)
param_grid = {"n_neighbors" : n_neighbors}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv = 3)
grid_search.fit(train_x, train_y)
print grid_search.best_params_['n_neighbors']
y_pred = grid_search.predict(test_x)
MSE_KK = np.mean((test_y - y_pred) ** 2.0)
print "The MSE of KNN = {}".format(MSE_KK)