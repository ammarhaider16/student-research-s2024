import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("../Datasets/Classification/default_credit_card_clients.csv")
df = df.rename(columns={"default payment next month" : "default"})

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

# Dropping missing values
df = df.dropna(subset=['default'])

# Random Forest Setup
X = df.drop('default', axis=1)
y = df['default']

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
y_pred_skLearn = model.predict(X_test)

accuracy_skl = accuracy_score(y_test, y_pred_skLearn)
print(f'Accuracy (sk-learn): {accuracy_skl:.2f}')


#XGBoost Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBRFClassifier(n_estimators=100, learning_rate=1, colsample_bynode=0.8, subsample=0.8, random_state=42))
])
model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)

accuracy_xgbrf = accuracy_score(y_test, y_pred_xgb)
print(f'Accuracy (XGBoost): {accuracy_xgbrf:.2f}')


# Write output data to output files
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
from utils.modelOutputToCSV import modelOutputToCSV

modelOneName = "sk-learn"
modelOneOutputList = y_pred_skLearn.astype(int).tolist()
modelTwoName = "XGBoost"
modelTwoOutputList = y_pred_xgb.tolist()

thisDirectory = "Classification"
thisFile = "default_credit_card_clients_RF"
filePath = f"/users/shaider/student-research-s2024/Data/{thisDirectory}/{thisFile}.csv"

# modelOutputToCSV(modelOneName,modelOneOutputList,modelTwoName,modelTwoOutputList,filePath)
from utils.dataAnalysis import handleAddClassificationAccuracy
handleAddClassificationAccuracy("RF",thisFile,accuracy_skl,accuracy_xgbrf)