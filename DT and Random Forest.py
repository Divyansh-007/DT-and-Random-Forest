# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:15:25 2020

@author: Dell
"""

# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier

# Importing the dataset
dataset = pd.read_csv("loan_data.csv")

# Exploring the dataset
dataset.head()
# Information about the dataset
dataset.info()

# Since the purpose column is catagorical we have to create dummy variables so that sklearn
# can undertsand them
cat_features = ['purpose']

# Crating the final dataset with dummy variables
final_dataset = pd.get_dummies(dataset,columns=cat_features,drop_first=True)
# Exploring the final dataset
final_dataset.head()
# Information about the final dataset
final_dataset.info()

X = final_dataset.drop('not.fully.paid',axis=1)
y = final_dataset['not.fully.paid']

# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training and Testing of DT Model
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)

# Evaluating the DT Model
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Training and Testing of Random Forest Model
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

# Evaluating the Random Forest Model
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

