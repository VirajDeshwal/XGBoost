#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:33:50 2018

@author: virajdeshwal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = pd.read_csv('Churn_Modelling.csv')
X = file.iloc[:,3:13].values
y = file.iloc[:,13].values
'''We have to encode the categorical data. As our Independent variable contains the string.'''

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

'''Let's import the XGBoost Library'''
#XGBoost model

from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(x_train, y_train)

#predictiong 

y_pred = model.predict(x_test)
#TRUE if prob>0.5|| false if prob <0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix

#show the true positive and false positive through the confusion matrix.
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n\n print the confusion matrix for true and false prediction rate.\n\n')
print(conf_matrix)


'''K-Fold Cross-Validation for much better accuracy'''
#Applying the K-Fold Cross Validation function
from sklearn.model_selection import cross_val_score
'''We will define a vector which will include the computed accuracies to evaluate our model.'''
accuracies =cross_val_score(estimator =model, X= x_train, y=y_train, cv=10)
print('\nThe average accuracy from 10 K-Fold is:', accuracies.mean())
print('\n\nThe Standard Deviation in the accuracies in 10 K-Fold is :',accuracies.std())

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy from XGBoost model is :', accuracy)

