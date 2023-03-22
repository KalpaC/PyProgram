# 测试dataframe 2023/3/20 20:11

import pandas as pd
import DataLayer

dl = DataLayer.DataLayer(DataLayer.read_csv('data.csv'),'60s')

df = dl.getResampleDataFrame()

print(df.head(10))

# 测试dataframe
diffs = pd.DataFrame(df['Temperature'])
diffs[1] = df['Temperature'].diff()[1:]
print(diffs.head())


