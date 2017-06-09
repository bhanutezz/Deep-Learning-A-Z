# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import os
# os.getcwd()
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

## Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
## Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer(activation function)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# with drop out
classifier.add(Dropout(rate = 0.1))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# in case if there are more dependent variables which has more than two categories
# use softmax activation function of sigmoid function and units are equal to no. of categories

## Compiling the ANN
# weights are uniformly distributed while creating NN in above steps but we have to add an algorithm
# which sets best weights to make NN more powerful, here stochastic gradient descent algorithms(optimizer ='adam')
# if your dependent variable has binary values, then logarithmic loss function is binary_crossentropy or
# if your dependent variables have more than two outcomes like three categories then loss function is categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# number of epocs(epoc is whole training set passed through the ANN): number of times we are training our ANN on whole training set
# from above stochastic gradient descent algorithm, we can see how ANN model is trained and how it 
# is imporving at each round i.e at each epoc in below step
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""
Predict if the customer with the following information will leave the bank
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN
"""K- Fold cross validation is a model from Scikit learn, where as our ANN model is developed using Keras. So there should be a way
to link these both. Keras provides a wrapper that wraps the scikit learn K-fold cross validation into Keras model"""

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# keras classifier expect a function as one of its first argument i.e build_fn. So we create a function below, which should include
# the steps to build the ANN as done above. So just copy the lines except fit/training line"""
# local classifier
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# Global classifier, which accepts build_classifier function and variable to fit the model which exempted in above function while copying from ANN template
# which also accepts batch_size and epochs as taken in fit function above
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# when to run the jobs on all cores of the processor, but it will not work as processor may not allow to use all cores
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regulariaztion to reduce overfitting if needed
# At each iteration some neurons of ANN are disabled randomly to prevent them from too dependent each other when they learn correlations
# Therefore by overwriting these neurons, the ANN learns several independent correlations in the data because each time there is not the same configuration of the neurons
# As neurons work more independtly, that prevents neurons larning too much that prevents over fitting 
# import new class "from keras.layers import Dropout"
# add this line classifier.add(Dropout(rate = 0.1)) for each layer

# Tuning the ANN
# parameter tuning, we have two types of parameters(weights and hyper parameters(no. of epochs))
# we can tune it using gridsearch, it is same as implementing cross_val_score above
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# as we have to tune the parameters, remove batch_size and epochs default values as they have to be tuned
classifier = KerasClassifier(build_fn = build_classifier)
# have to create dictionary that contains hyper parameters which have to be optimized
# when we want to tune hyper parameters in the architecture such as units, loss, activation,optimizer, metrics; we have to generalize them
# ** optimizer hyper parameter is already has default value 'adam' in previous implementation, as it is going to be tuned, optimizer has to be generalized not with tdefault value
# to do that, pass hyper parameter to the function and replace it as argument compile() function
# adam and rmsprop are stochastic gradient descent optimizers
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


