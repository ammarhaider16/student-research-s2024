import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("../Datasets/Classification/credit_customers.csv")

# Handle missing values -> drop columns with more than 90% missing values and replace other missing values with the average for the column

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

# Converting target values to numeric
df['class'] = df['class'].map({'good':1, 'bad':0})
df = df.dropna(subset=['class'])

# Random Forest Setup
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding for cat cols
cat_cols = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# SkLearn Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_skl = accuracy_score(y_test, y_pred)
print(f'Accuracy (sk-learn): {accuracy_skl:.2f}')


#XGBoost Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBRFClassifier(n_estimators=100, learning_rate=1, colsample_bynode=0.8, subsample=0.8, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_xgbrf = accuracy_score(y_test, y_pred)
print(f'Accuracy (XGBoost): {accuracy_xgbrf:.2f}')