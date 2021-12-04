# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset

dataset = pd.read_csv("Salary_data.csv")
# x : the dependent variable -> salary
# y : the independent variable -> years experience

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split the data into training dataset and test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


