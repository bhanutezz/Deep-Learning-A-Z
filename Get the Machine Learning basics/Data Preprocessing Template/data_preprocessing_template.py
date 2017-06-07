# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset, pandas for data analysis
dataset = pd.read_csv('Data.csv')
# all observations for all columns except last column, predictors
X = dataset.iloc[:, :-1].values
# column which has predicted values, response variables
y = dataset.iloc[:, -1].values

# split the dataset into tarining and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
# most of ML algorithms uses Euclidean distance =sqrt((x2-x1)^2 + (y2-y1)^2) to find distance between two points, 
# so both x and y coordinate observation points should be in same scale, otherwise one feature will dominate other(eg, salary dominates age in our example) 
# standardization and normalization two methods used for feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
