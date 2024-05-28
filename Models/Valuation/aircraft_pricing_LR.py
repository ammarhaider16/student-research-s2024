import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import re

# Load the dataset
dict = arff.load(open('../Datasets/Valuation/aircraft_pricing.arff'))

# Extract attribute names
title = [attr[0] for attr in dict['attributes']]

# Create DataFrame
df = pd.DataFrame(dict['data'])
df.columns = title


# Drop irrelevant columns
df = df[['Condition', 'Category', 'Year', 'Make', 'Model','Total_Seats','Price']]
df = df[df.ne('Not Listed').all(axis=1)]  # Filter out rows containing "Not Listed"
df = df[df.Year != "-"]

# Handle missing values
threshold = 0.9 * len(df)
cols_to_drop = df.columns[df.isnull().sum() > threshold]
df = df.drop(cols_to_drop, axis=1)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = ['Condition', 'Category', 'Make','Model']

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))

for col in cat_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

# Perform one-hot encoding for each categorical column
for col in cat_cols:
    one_hot_encoded = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop(col, axis=1, inplace=True)

# Converting string values to numbers
df['Price'] = df['Price'].str.strip()
df['Price'] = df['Price'].str.replace(",", "")
df['Price'] = df['Price'].astype(float)
df['Year'] = df['Year'].astype(float)

# Linear Regression Model setup using scikit-learn
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_skLearn = model.predict(X_test)
mse_skLearn = mean_squared_error(y_test, y_pred_skLearn)

print(f"Mean Squared Error (scikit-learn): {mse_skLearn}")

# Linear Regression Model setup usng numpy
X = df.drop('Price', axis=1)
y = df['Price'].values
X = np.hstack([np.ones((X.shape[0], 1)), X.values])


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
