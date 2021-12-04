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

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
