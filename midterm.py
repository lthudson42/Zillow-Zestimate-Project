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
data = read_csv('/Users/lthud/Downloads/ECO4444 Midterm/mid_term_dataset.csv', delimiter=',')

# Check for null values
data.isnull().sum()

# Scale price data
y = data.price/1000

# Store column names
columns = data.columns

# Loop through each column in the dataset, generate descriptive statistics, and histograms

for col in columns:
    print(f"\nDescriptive statistics for {col}:")
    
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

# Randomize data with seed = 1234
seed(1234)
data = data.sample(len(data))


# Linear Regression function definition
def model(features):

    for n in range(0, len(x_combos)):
        combo_list = list(x_combos[n])
        x = data[combo_list]
        x = sm.add_constant(x)
        poly = PolynomialFeatures(features)
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

# MODELS 

# PolynomialFeatures = 1

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

model(2)

# MODELS PolynomialFeatures = 3

model(3)
        
# MODELS PolynomialFeatures = 4

model(4)

# MODELS PolynomialFeatures = 5

model(5)

# Create a new variable

data['home_ratio'] = data.home_size / data.parcel_size

# Update columns variable
columns = data.columns

# Generate descriptive statistics
mean = round(np.mean(data.home_ratio), 2)
median = round(np.median(data.home_ratio), 2)
variance = round(np.var(data.home_ratio), 2)
sd = round(np.std(data.home_ratio), 2)
 
print(f"{data.home_ratio.name} - Mean = {mean}")
print(f"{data.home_ratio.name} - Median = {median}")
print(f"{data.home_ratio.name} - Variance = {variance}")
print(f"{data.home_ratio.name} - Standard Deviation = {sd}")
 
# Create a histogram
plt.hist(data.home_ratio, bins = 100)
plt.xlabel(data.home_ratio.name)
plt.ylabel('Number of Properties')
plt.title('Distribution of ' + data.home_ratio.name)
plt.xlim(0,1)
plt.show()

# Repeat training and testing process

x_combos = []

for n in range(1, 12):
    combos = combinations(['year', 'age', 'beds', 'baths', 
                           'home_size', 'parcel_size', 'pool', 'dist_cbd', 
                           'dist_lakes', 'x_coord', 'y_coord', 'home_ratio'], n)
    x_combos.extend(combos)
    
# MODELS 

# Poly = 1

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
        
# Poly = 2

model(2)
        
# Poly = 3

model(3)

# Poly = 4

model(4)

# Poly = 5

model(5)

# Test the best model on the full dataset
data = read_csv('/Users/lthud/Downloads/ECO4444 Midterm/mid_term_dataset.csv', delimiter=',')
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




