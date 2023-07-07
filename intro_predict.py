# %%
# pip install pandas_datareader

# %%
# pip install scikit-learn

# %%
# pip install yfinance

# %%
import os
import datetime as dt

import IPython.display

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf

import sklearn
import sklearn.linear_model
import sklearn.model_selection

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

yf.pdr_override()
symbol_aapl = 'AAPL'
symbol_meta = 'META'
symbol_gld = 'GLD'

start = dt.date(2014, 1, 1)
end = dt.date.today()

df_AAPL = pdr.get_data_yahoo(symbol_aapl, start, end)
df_META = pdr.get_data_yahoo(symbol_meta, start, end)
df_GLD = pdr.get_data_yahoo(symbol_gld, start, end)


df_AAPL.to_csv("data_stock/{}.csv".format(symbol_aapl))
df_META.to_csv("data_stock/{}.csv".format(symbol_meta))
df_GLD.to_csv("data_stock/{}.csv".format(symbol_gld))

df_AAPL['SMA'] = df_AAPL['Close'].rolling(window=14).mean()
df_AAPL['Close'].plot(figsize=(15,6), color="red")
df_AAPL['SMA'].plot(figsize=(15,6), color="green")
plt.show()

df_AAPL['change'] = (((df_AAPL['Close'] - df_AAPL['Open'])) / (df_AAPL['Open']) * 100)
df_META['change'] = (((df_META['Close'] - df_META['Open'])) / (df_META['Open']) * 100)
df_GLD['change'] = (((df_GLD['Close'] - df_GLD['Open'])) / (df_GLD['Open']) * 100)

df_AAPL.tail(2).round(2)
df_AAPL['Close'].plot(figsize=(15,6), color="red")
df_META['Close'].plot(figsize=(15,6), color="blue")
df_GLD['Close'].plot(figsize=(15,6), color="orange")
plt.show()

df_AAPL['change'].tail(100).plot(grid=True, figsize=(15,6), color="red")
df_META['change'].tail(100).plot(grid=True, figsize=(15,6), color="blue")
df_GLD['change'].tail(100).plot(grid=True, figsize=(15,6), color="orange")
plt.show()


df_AAPL['label'] = df_AAPL['Close'].shift(-30)
df_AAPL.tail(40)


# 機械学習(マシンラーニング)

# ラベル行を削除したデーターをXに代入
X = np.array(df_AAPL.drop(['label', 'SMA'], axis=1))
# 取りうる値の大小が著しく異なる特徴量を入れると結果が悪くなり、平均を引いて、標準偏差で割ってスケーリングする
X = sklearn.preprocessing.scale(X)

# 予測に使う過去30日間のデーター
predict_data = X[-30:]
# 過去30日を取り除いた入力データー
X = X[:-30]
y = np.array(df_AAPL['label'])
# 過去30日を取り除いた正解ラベル
y = y[:-30]

# 訓練データー80% 検証データー 20%に分ける
# 第一引数に入力データー、第２引数に正解ラベルの配列
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size = 0.2)

# 訓練データーを用いて学習する
lr = sklearn.linear_model.LinearRegression()
lr.fit(X_train,y_train)

# 検証データーを用いて検証してみる
accuracy = lr.score(X_test, y_test)
accuracy

# 予測する
predicted_data = lr.predict(predict_data)
predicted_data


df_AAPL['Predict'] = np.nan

last_date = df_AAPL.iloc[-1].name

one_day = 86400
next_unix = last_date.timestamp() + one_day

for data in predicted_data:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df_AAPL.loc[next_date] = np.append([np.nan]* (len(df_AAPL.columns)-1), data)

df_AAPL['Close'].plot(figsize=(15,6), color="green")
df_AAPL['Predict'].plot(figsize=(15,6), color="orange")
plt.show()


if df_AAPL['Predict'][-1] > df_AAPL['Close'][-31]:
    print('Buy using REST API')
else:
    print('Sell using REST API')


