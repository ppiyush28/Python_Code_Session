# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:47:50 2018

@author: DH384961
"""

"""
0 Regression Diagnostics
1 Unusual and Influential data
2 Tests on Normality of Residuals
3 Tests on Nonconstant Error of Variance
4 Tests on Multicollinearity
5 Tests on Nonlinearity
6 Model Specification
7 Issues of Independence
8 Summary
9 For more information
"""



# imports
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
#from statsmodels.stats.outliers_influence import variance_inflation_factor


"""
3. Example: Advertising Data¶
Let's take a look at some data, ask some questions about that data, 
and then use linear regression to answer those questions!
"""

# read data into a DataFrame
data = pd.read_csv('E:/Class/Data/Advertising.csv', index_col=0)
data.head()

# shape of the DataFrame
data.shape

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=7, aspect=0.7)

# Check VIF
#vif = [variance_inflation_factor(data.drop(["sales"],axis=1).values, i) for i in range(data.drop(["sales"],axis=1).shape[1])]
#check correlation
corr = (data.drop(["sales"], axis = 1).corr())
corr2 = data.corr()
### STATSMODELS ###

# The model coefficients for the advertising data
# create a fitted model
lm1 = smf.ols(formula='sales ~ TV', data=data).fit()

# print the coefficients
lm1.params  


### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV']
X = data[feature_cols]
y = data.sales

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print(lm2.intercept_)
print(lm2.coef_)


### STATSMODELS ###

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50]})

# predict for a new observation
lm1.predict(X_new)

### SCIKIT-LEARN ###

# predict for a new observation
lm2.predict(50)

# Plotting the Least Squares Line
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')


#Confidence in our Model
### STATSMODELS ###

# print the confidence intervals for the model coefficients
lm1.conf_int() 

### STATSMODELS ###

# print the p-values for the model coefficients
lm1.pvalues

# How Well Does the Model Fit the data?
### STATSMODELS ###

# print the R-squared value for the model
lm1.rsquared

### SCIKIT-LEARN ###

# print the R-squared value for the model
lm2.score(X, y)

#Multiple Linear Regression
### STATSMODELS ###

# create a fitted model with all three features
lm1 = smf.ols(formula='sales ~ TV + radio + newspaper', data=data).fit()

# print the coefficients
lm1.params

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print(lm2.intercept_)
print(lm2.coef_)

# pair the feature names with the coefficients
list(zip(feature_cols, lm2.coef_))

### STATSMODELS ###

# print a summary of the fitted model
lm1.summary()


#Feature Selection¶
### STATSMODELS ###

# only include TV and Radio in the model

# instantiate and fit model
lm1 = smf.ols(formula='sales ~ TV + radio', data=data).fit()

# calculate r-square 
lm1.rsquared

# add Newspaper to the model (which we believe has no association with Sales)
lm1 = smf.ols(formula='sales ~ TV + radio + newspaper', data=data).fit()
lm1.rsquared


# Metrics
# define true and predicted response values
y_true = [100, 50, 30, 20]
y_pred = [90, 50, 50, 30]

# calculate MAE, MSE, RMSE
print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

#Model Evaluation Using Train/Test Split
# include Newspaper
X = data[['TV', 'radio', 'newspaper']]
y = data.sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state=1)

# Instantiate model
lm2 = LinearRegression()

# Fit Model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# exclude Newspaper
X = data[['TV', 'radio']]
y = data.sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

# Instantiate model
lm2 = LinearRegression()

# Fit model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))

#Handling Categorical Features with Two Categories

# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large
data['Size'] = 'small'

# Series.loc is a purely label-location based indexer for selection by label
data.loc[mask_large, 'Size'] = 'large'
data.head()

# create a new Series called Size_large
data['Size_large'] = data.Size.map({'small':0, 'large':1})
data.head()

# create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'Size_large']
X = data[feature_cols]
y = data.sales


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

# instantiate
lm2 = LinearRegression()
# fit
lm2.fit(X_train, y_train)

y_pred = lm2.predict(X_test)

mse =metrics.mean_squared_error(y_test,y_pred)
np.sqrt(mse)
metrics.r2_score(y_test,y_pred)

feature_cols = ['TV', 'newspaper', 'Size_large']

# print coefficients
list(zip(feature_cols, lm2.coef_))

# Handling Categorical Features with More than Two Categories
# set a seed for reproducibility
np.random.seed(123456)

# assign roughly one third of observations to each group
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
# Series.loc is a purely label-location based indexer for selection by label
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()

# create three dummy variables using get_dummies
pd.get_dummies(data.Area, prefix='Area').head()

# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]
area_dummies.head()

# concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()


# create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'Size_large', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X_train, y_train)

# print the coefficients
list(zip(feature_cols, lm2.coef_))


####################################

# Regression Using Support Vector Machines
# Note for best result perform GRID Search
from sklearn.svm import SVR
feature_cols = ['TV', 'radio', 'newspaper', 'Size_large', 'Area_suburban', 'Area_urban']
#feature_cols = ["TV"]
X = data[feature_cols]
y = data.sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)

print(metrics.mean_absolute_error(y_test, y_rbf))
print(metrics.mean_squared_error(y_test, y_rbf))
print(np.sqrt(metrics.mean_squared_error(y_test, y_rbf)))

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [np.mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

coefficient_of_determination(y_test,y_rbf)
metrics.r2_score(y_test,y_rbf)

# Regression Using Random Forest
# Note for best result perform GRID Search
from sklearn.ensemble import RandomForestRegressor as RFR
RFR_model = RFR()
RFR_model.fit(X_train,y_train)
y_RFR = RFR_model.predict(X_test)

coefficient_of_determination(y_test,y_RFR)
metrics.r2_score(y_test,y_RFR)

print(metrics.mean_absolute_error(y_train, y_RFR))
print(metrics.mean_squared_error(y_test, y_RFR))
print(np.sqrt(metrics.mean_squared_error(y_test, y_RFR)))


