import arff
import pandas as pd

dict = arff.load(open('../Datasets/Valuation/toronto_apartment_rental_prices.arff'))

title = []
for list in dict.get('attributes'):
    title.append(list[0])


df = pd.DataFrame(dict.get("data"))
df.columns = title
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=['Address'])

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

# Trimming price
df['Price'] = df['Price'].str.replace(",","").astype(float)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Linear Regression Model setup usng sk-learn
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred_skLearn = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_skLearn)

print(f"Mean Squared Error (scikit-learn): {mse}")


# Linear Regression Model setup using numpy
X = df.drop('Price', axis=1)
y = df['Price'].values
X = np.hstack([np.ones((X.shape[0], 1)), X.values])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_T = X_train.T
theta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train)

y_pred_numPy = X_test.dot(theta)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred_numPy)

print(f"Mean Squared Error (numpy): {mse}")