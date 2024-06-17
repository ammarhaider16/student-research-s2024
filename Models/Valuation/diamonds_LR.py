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

#OneHotEncoding
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot_encoded = onehot_encoder.fit_transform(df[cat_cols])
categories = onehot_encoder.categories_
feature_names = []
for i, col in enumerate(cat_cols):
    for category in categories[i]:
        feature_names.append(f"{col}_{category}")
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=feature_names)
df = pd.concat([df, onehot_encoded_df], axis=1)
df.drop(cat_cols, axis=1, inplace=True)

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

X_train_T = X_train.T
theta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train)

y_pred_numPy = X_test.dot(theta)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred_numPy)

print(f"Mean Squared Error (numpy): {mse}")
 

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
modelTwoOutputList = y_pred_numPy.tolist()

thisDirectory = "Valuation"
thisFile = "diamonds_LR"
filePath = f"/users/shaider/student-research-s2024/Data/{thisDirectory}/{thisFile}.csv"

modelOutputToCSV(modelOneName,modelOneOutputList,modelTwoName,modelTwoOutputList,filePath)