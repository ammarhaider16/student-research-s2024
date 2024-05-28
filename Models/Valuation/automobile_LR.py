import pandas as pd


df = pd.read_csv("../Datasets/Valuation/automobile.csv")
df = df.drop(columns = ['symboling', 'normalized-losses','num-of-doors'])

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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Drop cat cols
df = df.drop(columns=cat_cols)

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
y = df['price'].values
X = np.hstack([np.ones((X.shape[0], 1)), X.values])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_T = X_train.T
theta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train)

y_pred_numPy = X_test.dot(theta)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred_numPy)

print(f"Mean Squared Error (numpy): {mse}")
