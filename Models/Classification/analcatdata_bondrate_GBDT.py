import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

dict = arff.load(open('../Datasets/Classification/analcatdata_bondrate.arff'))

title = []
for list in dict.get('attributes'):
    title.append(list[0])


df = pd.DataFrame(dict.get("data"))
df.columns = title
df = df.drop(columns=['City'])

# Handle missing values 
threshold = 0.9* len(df)
cols_to_drop = df.columns[df.isnull().sum() > threshold]
df = df.drop(cols_to_drop, axis=1)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))

for col in cat_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

df['binaryClass'] = df['binaryClass'].map({'N': 0, 'P': 1})

# GBDT Setup
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


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
thisFile = "analcatdata_bondrate_GBDT"
filePath = f"/users/shaider/student-research-s2024/Data/{thisDirectory}/{thisFile}.csv"

modelOutputToCSV(modelOneName,modelOneOutputList,modelTwoName,modelTwoOutputList,filePath)