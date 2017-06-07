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
# taking care if missing data
from sklearn.preprocessing import Imputer
# press i to object inspector after typing class name ex.'Imputer + ctrl +i', you will get constructor details
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# fit the imputer to X dataset and set NaN for columns which have missing data, 1:3 - 3 is upper bound which is exluded
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# above line can also be used as imputer.fit_transform(X[:, 1:3])
#categorical variables, country and purchased columns in dataset have to be converted into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# by above line country column values are changed to 0,1,2 numbers but there is a problem when it compares each other ass they are numbers
# usually from the values Germany is greater than France and France greater than spain which is not useful in machine learning algorithms
# t eliminate this problem, there is concept called Dummy Encoding/Variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#label encoder for response variable, i.e purchased
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

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
