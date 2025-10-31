# Zillow-Zestimate-Project
ECO4444 Mid-Term project
Authors: Liam Hudson and Shyam Patel

To view a comprehensive report of this project, click this [link](https://docs.google.com/document/d/1wsAdmPym7Qb1zBZzKSwvG4mbaPXadfTWoNlhcetCst8/edit?usp=sharing).

In this project, we explore a dataset of property sales data with multiple variables.

* 'price': the final sales price of the property
* 'year': the year in which the sale occurred
* 'age': the age of the property
* 'beds': number of bedrooms
* 'baths': number of bathrooms
* 'home_size': in square feet, size of the home
* 'parcel_size': in square feet, size of the entire property (including home_size)
* 'pool': dummy variable (0/1)
* 'dist_cbd': in meters, distance from the nearest central business district
* 'dist_lakes': in meters, distance from the nearest lake
* 'x_coord': geographic location (horizontal axis)
* 'y_coord': geographic location (vertical axis)

Our goal is to train and test (using 10-fold cross validation) a multitude of linear regression models with various specifications (i.e., logarithmic, quadratic, polynomial, etc.) to identify
the 'best' model. A model is deemed the 'best' if it has the lowest training mean squared error (MSE). Once we have identified our best model, we use it to predict the full dataset. Lastly,
we introduce a validation dataset to estimate the results.

Factors to consider when creating a linear regression model:

* Complexity
* Overfitting
* Multicollinearity
