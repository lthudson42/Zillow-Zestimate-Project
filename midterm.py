#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:48:31 2025

@author: Liam Hudson and Shyam Patel
"""

import pandas as pd
from pandas import read_csv
import numpy as np
from numpy.random import seed

from itertools import combinations
import statsmodels.api as sm
from statsmodels.api import add_constant

import statsmodels.formula.api as smf

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/mid_term_dataset.csv', delimiter=',')

# Check for null values
data.isnull().sum()

# Store column names
columns = data.columns

# Loop through each column in the dataset, generate descriptive statistics, and histograms

for col in columns:
    print(f"Descriptive statistics for {col}:\n")
    
    mean = round(np.mean(data[col]), 2)
    median = round(np.median(data[col]), 2)
    variance = round(np.var(data[col]), 2)
    sd = round(np.std(data[col]), 2)
    
    print(f"{col} - Mean = {mean}")
    print(f"{col} - Median = {median}")
    print(f"{col} - Variance = {variance}")
    print(f"{col} - Standard Deviation = {sd}")
    
    # Create a histogram for each variable
    plt.hist(data[col], bins = 100)
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Number of Properties')
    plt.title('Distribution of ' + col.replace('_', ' ').title())
    plt.show()
    
    
# Correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

x_combos = []

for n in range(1, 11):
    combos = combinations(['year', 'age', 'beds', 'baths', 
                           'home_size', 'parcel_size', 'pool', 'dist_cbd', 
                           'dist_lakes', 'x_coord', 'y_coord'], n)
    x_combos.extend(combos)

# Scale down price data to make related calculations more manageable
y = data.price/1000

# Randomize data with seed = 1234
seed(1234)
data = data.sample(len(data))


# MODELS PolynomialFeatures = 1

models = {}
mse = {}

for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x = data[combo_list]
    model = LinearRegression()
    cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
    mse[str(combo_list)] = np.mean(cv_scores['test_score'])

print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse)
for possibles, i in mse.items():
    if i == -min_mse:
        print("The Combination of Variables:", possibles)

# MODELS PolynomialFeatures = 2

for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x = data[combo_list]
    y = data['price']
    x = sm.add_constant(x)
    poly = PolynomialFeatures(2)
    poly_x = poly.fit_transform(x)
    model = sm.OLS(y,poly_x)
    results = model.fit()
    mse = results.mse_resid
    models[str(combo_list)] = mse
    
min_mse = min(models.values())
print("The Minimum MSE:", min_mse)
for possibles, i in models.items():
    if i == min_mse:
        print("The variable combinations:", possibles)

# MODELS PolynomialFeatures = 3

for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x = data[combo_list]
    y = data['price']
    x = sm.add_constant(x)
    poly = PolynomialFeatures(3)
    poly_x = poly.fit_transform(x)
    model = sm.OLS(y,poly_x)
    results = model.fit()
    mse = results.mse_resid
    models[str(combo_list)] = mse
    
min_mse = min(models.values())
print("The Minimum MSE:", min_mse)
for possibles, i in models.items():
    if i == min_mse:
        print("The variable combinations:", possibles)


# Test the best model on the full dataset
data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/mid_term_dataset.csv', delimiter=',')
data['price'] = data['price']/1000

x = data['home_size']
poly = PolynomialFeatures(2)
poly_x = poly.fit_transform(x)

x = pd.DataFrame(poly.fit_transform(x), columns=poly.get_feature_names_out(x.columns))
y = data['price']

best_model = sm.OLS(y, x)
results = best_model.fit()
print(results.summary())

# Introduce new data

new_data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/', delimiter=',')
new_data['price'] = new_data['price']/1000

x = new_data[['''best variable combos''']]
poly = PolynomialFeatures('''polynomial feature''')
poly_x = poly.fit_transform(x)




