import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

dict = arff.load(open('../Datasets/Classification/creditcard_fraud_detection.arff'))


title = []
for list in dict.get('attributes'):
    title.append(list[0])


df = pd.DataFrame(dict.get("data"))
df.columns = title
df = df.drop(columns=["Time"])

# No missing values in dataset

# Converting target values to numeric
df['Class'] = df['Class'].map({'0':1, '1':0})
df = df.dropna(subset=['Class'])

# GBDT Setup
X = df.drop('Class', axis=1)
y = df['Class']
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
y_pred_xgb = model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy (XGBoost): {accuracy_xgb}")


# GBDT implemented with LightGBM
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier())
])
model.fit(X_train, y_train)
y_pred_lgbm = model.predict(X_test)

accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"Accuracy (LightGBM): {accuracy_lgbm}")


# Write output data to output files
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)


from utils.modelOutputToCSV import modelOutputToCSV
modelOneName = "XGBoost"
modelOneOutputList = y_pred_xgb.tolist()
modelTwoName = "LightGBM"
modelTwoOutputList = y_pred_lgbm.tolist()

thisDirectory = "Classification"
thisFile = "creditcard_fraud_detection_GBDT"
filePath = f"/users/shaider/student-research-s2024/Data/{thisDirectory}/{thisFile}.csv"

# modelOutputToCSV(modelOneName,modelOneOutputList,modelTwoName,modelTwoOutputList,filePath)
from utils.dataAnalysis import handleAddClassificationAccuracy
handleAddClassificationAccuracy("GBDT",thisFile,accuracy_xgb,accuracy_lgbm)