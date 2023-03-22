# test 2023/3/4 22:30
from DataLayer import *
from CAG import *


df = read_csv('data.csv')
dl = DataLayer(df, '60s')
cag = CAG(dl)
cIMFs, res = cag.getCIMFs('Temperature')

cimf = cIMFs[3]
print(type(cimf))

test = pd.Series(cimf)
df = pd.DataFrame(test,columns=[0])
for i in range(1,4):
    df[i] = df[i-1].diff()
print(df.head())
print(df.shape[0])
# arima = MyARIMA(test)
# arima.predict_forecast(3 * len(test) // 4, len(test) + 20)
