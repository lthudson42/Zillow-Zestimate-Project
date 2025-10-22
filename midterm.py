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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt



# Load data
data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/mid_term_dataset.csv', delimiter=',')

# Check for null values
data.isnull().sum()

# Scale price data
y = data.price/1000
data.price = y

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


# Using itertools, create a combinations object of all possible variable combos.
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
        
        # Polynomial specifications
        poly = PolynomialFeatures(features)
        poly_x = poly.fit_transform(x)
        
        # Fit model
        model = sm.OLS(y,poly_x)
        results = model.fit()
        
        # Record MSE
        mse = results.mse_resid
        models[str(combo_list)] = mse
        
    # Print minimum average test MSE
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
        
'''Minimum Average Test MSE: 20,484
    Variables: pool'''

# PolynomialFeatures = 2

model(2)

'''Minimum Average Test MSE: 20,462
    Variables: year, age, home_size, dist_lakes'''

# PolynomialFeatures = 3

model(3)

'''Minimum Average Test MSE: 20,361
    Variables: year, beds, baths, home_size, parcel_size, pool, dist_cbd,
    dist_lakes, x_coord, y_coord'''
        
# PolynomialFeatures = 4

model(4)

'''Minimum Average Test MSE: 19,946
    Variables: year, age, beds, baths, home_size, parcel_size, dist_cbd, dist_lakes,
    x_coord, y_coord'''

# PolynomialFeatures = 5

model(5)

'''Minimum Average Test MSE: 19,271
    Variables: year, age, beds, baths, home_size, parcel_size, dist_cbd, dist_lakes,
    x_coord, y_coord'''



# Create a new variable: home_ratio

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

# Re-randomize data with seed = 1234
seed(1234)
data = data.sample(len(data))

# Re-initialize x_combos with new variable
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
        
'''Minimum Average Test MSE: 20,484
    Variables: pool'''
        
# Poly = 2

model(2)

'''Minimum Average Test MSE: 20,460
    Variables: year, bath, home_size, dist_lakes, home_ratio'''
        
# Poly = 3

model(3)

'''Minimum Average Test MSE: 20,345
    Variables: year, age, beds, bath, home_size, parcel_size, pool, dist_cbd, dist_lakes,
    x_coord, y_coord'''

# Poly = 4

model(4)

'''Minimum Average Test MSE: 19,869
    Variables: year, age, beds, bath, home_size, parcel_size, dist_cbd, 
    dist_lakes, x_coord, y_coord, home_ratio'''

# Poly = 5

model(5)

'''
Minimum Average Test MSE: 19,135
Variables: year, age, beds, baths, home_size, parcel_size, pool, dist_cbd, dist_lakes, x_coord, y_coord
'''


# New variable: two_baths (does the property have at least 2 bathrooms?)

data.loc[:, 'two_baths'] = 0 # Use .loc to avoid creating a copy of a dataframe
for i in range(0, len(data.baths)):
    if data.loc[i, 'baths'] >= 2.0:
        data.loc[i, 'two_baths'] = 1
print(data.two_baths)

# Update columns variable
columns = data.columns

# Re-randomize data with seed = 1234
seed(1234)
data = data.sample(len(data))

# Re-initialize x_combos with new variables
x_combos = []

for n in range(1, 13):
    combos = combinations(['year', 'age', 'beds', 'baths', 
                           'home_size', 'parcel_size', 'pool', 'dist_cbd', 
                           'dist_lakes', 'x_coord', 'y_coord', 'home_ratio', 'two_baths'], n)
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

# Print minimum average test MSE
print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse)
for possibles, i in mse.items():
    if i == -min_mse:
        print("The Combination of Variables:", possibles)

'''Minimum Average Test MSE: 20,480
    Variables: home_size, parcel_size'''
    
# Poly = 2

model(2)

'''Minimum Average Test MSE: 20,440
    Variables: year, age, home_size, pool, dist_cbd, dist_lakes, two_baths'''

# Poly = 3

model(3)
'''Minimum Average Test MSE: 20,310
    Variables: year, age, home_size, pool, dist_cbd, dist_lakes, two_baths'''


''' 
At this point, these increasing polynomial features are becoming very computationally challenging 
(est. time of completion being over 60 minutes). From here, we can select our best model.
 '''
 

# Raising all variables (exclusding dummy variables) to non-integer exponents

base_features = ['year', 'age', 'beds', 'baths', 'home_size', 'parcel_size', 'dist_cbd',
                 'dist_lakes', 'x_coord', 'y_coord', 'home_ratio']

models = {}
mse = {}

# Create a range of 25 integers, non-inclusive (0 to 24)
i = np.arange(25)
# Divide each integer by 25
# This will create an array of evenly-spaced decimal values between 0 and 1
z = i/len(i)

for feature in base_features:
    for n in range(25):
        data[f'{feature}_exp'] = np.power(data[f'{feature}'],z[n])
        x = np.transpose(np.array([data[feature], data[f'{feature}_exp']]))
        x = pd.DataFrame(x, columns = [feature, f'{feature}_exp'])
        model = LinearRegression()
        cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
        mse[f'{feature}_{z[n]}'] = np.mean(cv_scores['test_score'])


# Print minimum average test MSE
print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(3))
for exponents, i in mse.items():
    if i == -min_mse:
        print("The Associated Exponent Values:", exponents)
        
'''
Minimum average test MSE: 20,480
Variable/exponent: 'parcel_size' raised to the 0.0 power
'''

        
# Now, let's try exponents between 1 and 2

models = {}
mse = {}

# Create a range of 25 integers, non-inclusive (0 to 24)
i = np.arange(25)
# Divide each integer by 25 and shift the values by 1
# This will create an array of evenly-spaced decimal values between 1 and 2
z = i/len(i) + 1

for feature in base_features:
    for n in range(25):
        data[f'{feature}_exp'] = np.power(data[f'{feature}'],z[n])
        x = np.transpose(np.array([data[feature], data[f'{feature}_exp']]))
        x = pd.DataFrame(x, columns = [feature, f'{feature}_exp'])
        model = LinearRegression()
        cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
        mse[f'{feature}_{z[n]}'] = np.mean(cv_scores['test_score'])


# Print minimum average test MSE
print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(3))
for exponents, i in mse.items():
    if i == -min_mse:
        print("The Associated Exponent Values:", exponents)     


'''
Minimum average test MSE: 20,480
Variable/exponent: parcel_size raised to the 1.0 power
'''

# 2 to 3

models = {}
mse = {}

i = np.arange(25)
z = i/len(i) + 2

for feature in base_features:
    for n in range(25):
        data[f'{feature}_exp'] = np.power(data[f'{feature}'],z[n])
        x = np.transpose(np.array([data[feature], data[f'{feature}_exp']]))
        x = pd.DataFrame(x, columns = [feature, f'{feature}_exp'])
        model = LinearRegression()
        cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
        mse[f'{feature}_{z[n]}'] = np.mean(cv_scores['test_score'])


# Print minimum average test MSE
print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(3))
for exponents, i in mse.items():
    if i == -min_mse:
        print("The Associated Exponent Values:", exponents)  
        
        
'''
Minimum average test MSE: 20,479
Variable/exponent: parcel_size raised to the 2.96 power
'''

# 3 to 4

models = {}
mse = {}

i = np.arange(25)
z = i/len(i) + 3

for feature in base_features:
    for n in range(25):
        data[f'{feature}_exp'] = np.power(data[f'{feature}'],z[n])
        x = np.transpose(np.array([data[feature], data[f'{feature}_exp']]))
        x = pd.DataFrame(x, columns = [feature, f'{feature}_exp'])
        model = LinearRegression()
        cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
        mse[f'{feature}_{z[n]}'] = np.mean(cv_scores['test_score'])


# Print minimum average test MSE
print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(3))
for exponents, i in mse.items():
    if i == -min_mse:
        print("The Associated Exponent Values:", exponents)  

'''
Minimum average test MSE: 20,479
Variable/exponent: parcel_size raised to the 3.72 power
'''

'''
From this pattern, we can see that the minimum average test MSE is not changing
all too much, no matter what non-integer exponents we are raising the features to.
'''



'''
We just executed a lot of modeling frameworks. 

To recap:
    * Modeled base dataset (no additional variables) and ran 10-fold cross-validation
      regressions with polynomial features up to 5
    * Created two new variables: home_ratio and two_baths
        * Ran 10-fold cross validation regressions on the new dataset with polynomial 
          features up to 5 and 3, respectively
    * Ran 10-fold cross-validation regressions on base features (excluding dummy variables)
      with non-integer exponents between 0 and 4 in increments of 0.04
      

With all these models, it is time to select the best one.

LOWEST AVERAGE TEST MSE: 19,135
VARIABLES (in dataset with just home_ratio): year, age, beds, baths, home_size, 
parcel_size, pool, dist_cbd, dist_lakes, x_coord, y_coord
Polynomial Features: 5
'''


# Use the best model to predict the full dataset

data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/mid_term_dataset.csv', delimiter=',')
data['price'] = data['price']/1000
data['home_ratio'] = data.home_size / data.parcel_size # must create home_ratio variable as best_model was trained on the dataset with it

x = data[['year', 'age', 'beds', 'baths', 'home_size', 'parcel_size', 'pool', 'dist_cbd', 'dist_lakes', 'x_coord', 'y_coord']]
poly = PolynomialFeatures(5)
poly_x = poly.fit_transform(x) # look at columns (x) and map data to all possible combinations (with specified degrees up to 5)

x = pd.DataFrame(poly.fit_transform(x), columns=poly.get_feature_names_out(x.columns))
y = data['price']

best_model = sm.OLS(y, x)
results = best_model.fit()
print(results.summary()) # may take a minute to load
print(results.rsquared) # 0.9096848858299772
predictions = results.predict(x)


# Introduce new data (validation set)

new_data = read_csv('/Users/liamhudson/Downloads/ucf_classes/eco444_fall_2025/python/data/val_set.csv', delimiter=',')
new_data['price'] = new_data['price']/1000 
new_data['home_ratio'] = new_data.home_size / new_data.parcel_size

# Create new poly transformed object to avoid re-fitting of the original
x_new = new_data[['year', 'age', 'beds', 'baths', 'home_size', 'parcel_size', 
                  'pool', 'dist_cbd', 'dist_lakes', 'x_coord', 'y_coord']]
poly_new = PolynomialFeatures(5)
poly_x_new = poly.transform(x_new) # .transform() applies the same mapping that best_model was trained on

val_predictions = results.predict(poly_x_new) # obtain predictions
