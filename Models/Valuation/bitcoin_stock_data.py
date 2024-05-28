import arff
import pandas as pd

dict = arff.load(open('../Datasets/Valuation/bitcoin_stock_data.arff'))
print(type(dict))

title = []
for list in dict.get('attributes'):
    title.append(list[0])


df = pd.DataFrame(dict.get("data"))
df.columns = title
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df)
