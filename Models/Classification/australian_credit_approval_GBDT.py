import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('../Datasets/Classification/australian.dat', sep=" ")


# Handle missing values -> drop columns with more than 90% missing values and replace other missing values with the average for the column

threshold = 0.9* len(df)
cols_to_drop = df.columns[df.isnull().sum() > threshold]
df = df.drop(cols_to_drop, axis=1)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))

for col in cat_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

# Dropping NaN values in target col
df = df.dropna(subset=['CLASS'])

# GBDT Setup
X = df.drop('CLASS', axis=1)
y = df['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# One-hot encoding for cat cols
cat_cols = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)


# GBDT implemented with XGB
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred)
print(f"Accuracy (XGBoost): {accuracy_xgb}")


# GBDT implemented with LightGBM
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_lgbm = accuracy_score(y_test, y_pred)
print(f"Accuracy (LightGBM): {accuracy_lgbm}")