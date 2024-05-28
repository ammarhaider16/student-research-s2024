import arff
import pandas as pd


# Set up the DataFrame

dict = arff.load(open('../Datasets/Valuation/3000_stock_financials.arff'))

print(dict)

title = []
for list in dict.get('attributes'):
    title.append(list[0])

df = pd.DataFrame(dict.get("data"))
df.columns = title
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

print(df)