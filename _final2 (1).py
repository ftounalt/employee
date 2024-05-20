#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pickle


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_

model_gb_tuned = GradientBoostingClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample']
)

model_gb_tuned.fit(x_train, y_train)
predictions_tuned_gb = model_gb_tuned.predict(x_test)

training_score = model_gb_tuned.score(x_train, y_train)
test_score = model_gb_tuned.score(x_test, y_test)





pickle.dump(model_gb_tuned,open('gb_model.pkl','wb'))




