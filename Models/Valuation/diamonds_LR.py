import pandas as pd
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("../Datasets/Valuation/diamonds.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values -> drop columns with more than 50% missing values and replace other missing values with the average for the column

threshold = 0.9* len(df)
cols_to_drop = df.columns[df.isnull().sum() > threshold]
print("Dropped columns: "+cols_to_drop)
df = df.drop(cols_to_drop, axis=1)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))

for col in cat_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

df = df.drop(columns=cat_cols)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Linear Regression Model setup usng sk-learn
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred_skLearn = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_skLearn)

print(f"Mean Squared Error (scikit-learn): {mse}")


# Linear Regression Model setup using numpy
X = df.drop('price', axis=1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a column of ones to the features for the intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Ridge regression using numpy
alpha = 1.0  # Regularization strength

# Calculate the coefficients using the ridge regression formula
X_train_T = X_train.T
theta = np.linalg.inv(X_train_T.dot(X_train) + alpha * np.identity(X_train.shape[1])).dot(X_train_T).dot(y_train)

# Make predictions on the test data
y_pred_ridge_numpy = X_test.dot(theta)

# Calculate the mean squared error
mse_ridge_numpy = mean_squared_error(y_test, y_pred_ridge_numpy)

print(f"Mean Squared Error (numpy): {mse_ridge_numpy}")
 

# Write output data to output files
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from utils.modelOutputToCSV import modelOutputToCSV
modelOneName = "sk-learn"
modelOneOutputList = y_pred_skLearn.tolist()
modelTwoName = "numpy"
modelTwoOutputList = y_pred_ridge_numpy.tolist()

thisDirectory = "Valuation"
thisFile = "diamonds_LR"
filePath = f"/users/shaider/student-research-s2024/Data/{thisDirectory}/{thisFile}.csv"

modelOutputToCSV(modelOneName,modelOneOutputList,modelTwoName,modelTwoOutputList,filePath)