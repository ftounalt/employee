#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pickle


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


# In[2]:


# Read dataset to pandas dataframe
dataset = pd.read_csv('Employee.csv') 

# Create an instance of LabelEncoder
le = LabelEncoder()

# Fit and transform the 'Education' column
dataset['Education'] = le.fit_transform(dataset['Education'])
dataset['City'] = le.fit_transform(dataset['City'])
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['EverBenched'] = le.fit_transform(dataset['EverBenched'])





X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train,x_test,y_train,y_test=train_test_split(X,y)

svm = SVC(kernel='rbf' , gamma=0.1 , C=10)
svm.fit(X_train, y_train)
y_preds = svm.predict(X_test)





pickle.dump(svm,open('gb_model.pkl','wb'))




