# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:59:38 2019

@author: b03881
"""

# Import libraries
import pandas as pd
import numpy as np

# Import data
dataset = pd.read_csv(r"U:\Nouveau dossier\data_scentist\data\mldata\Churn_Modelling.csv")
print (dataset.head())

X = dataset.iloc[:, 3:13]
print (X.head())

y = dataset.iloc[:, 13]
print (y.head())

# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

#http://www.xavierdupre.fr/app/papierstat/helpsphinx/notebooks/artificiel_category_2.html
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))

X = preprocess.fit_transform(X)
print(dataset['Geography'].head())

X = np.delete(X, [0,3], 1)
print (X.head())

# Split in train/test
y = y.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print (y_test)


