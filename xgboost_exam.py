# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:40:38 2020

@author: suyog
"""

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onthotencoder = OneHotEncoder(categorical_features=[1])
X = onthotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Splitting dataset to the test and train data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#fitting Xboost for Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

#predict the test set results
y_pred = classifier.predict(X_test)

#make the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)






