from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_raw = np.load('rawdata/npy/8h_2days_fill.npy', allow_pickle=True)
print(len(data_raw[1][1]))
datar = data_raw[98][1]
data = pd.Series(datar)
'''
discfile = 'E:/data_test.xls'
data = pd.read_excel(discfile,index_col=0)
data=data['number']
data.head()
'''
plt.figure(1,figsize=(12, 8), dpi=300)

plt.plot(data)


# Using the first difference, 12 - step difference processing time series
diff_1 = data.diff(1)
diff1 = diff_1.dropna()
diff1_144_1 = diff_1-diff_1.shift(144)
diff1_144 = diff1_144_1.dropna()
#print(diff1_144_1)

fig1 = plt.figure(figsize=(12,8))
ax1=fig1.add_subplot(111)
sm.graphics.tsa.plot_acf(diff1_144,lags=40,ax=ax1)
fig2 = plt.figure(figsize=(12,8))
ax2=fig2.add_subplot(111)
sm.graphics.tsa.plot_pacf(diff1_144,lags=40, ax=ax2)


# arma_mod01 = sm.tsa.ARMA(diff1_144,(0,1)).fit()
# print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
# arma_mod10 = sm.tsa.ARMA(diff1_144,(1,0)).fit()
# print(arma_mod10.aic,arma_mod10.bic,arma_mod10.hqic)
# arma_mod60 = sm.tsa.ARMA(diff1_144,(6,0)).fit()
# print(arma_mod60.aic,arma_mod60.bic,arma_mod60.hqic)
arma_mod61 = sm.tsa.ARMA(diff1_144,(6,1)).fit()
print(arma_mod61.aic,arma_mod61.bic,arma_mod61.hqic)

resid = arma_mod61.resid

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

print(sm.stats.durbin_watson(arma_mod61.resid.values))

r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
d = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(d, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_data=arma_mod61.predict(0,-1,dynamic=False)


# 144 differential restore
diff1_144_shift=diff_1.shift(144)
# print('print diff1_144_shift')
print(diff1_144_shift)
diff_recover_144=predict_data.add(diff1_144_shift)
# First difference reduction
diff1_shift=data.shift(1)
diff_recover_1=diff_recover_144.add(diff1_shift)
diff_recover_1=diff_recover_1.dropna() # 最终还原的预测值
print('predicted value')
print(diff_recover_1)

plt.figure(2,figsize=(10, 5), dpi=300)
plt.xticks(rotation=45)
plt.plot(diff_recover_1)
plt.plot(data[1800:2040])

'''
# Actual value, predicted value and differential predicted value are plotted
fig, ax = plt.subplots(figsize=(12, 8))
ax = data.ix['2017-04-01':].plot(ax=ax)
ax = diff_recover_1.plot(ax=ax)
fig = arma_mod61.plot_predict('2017/4/2 23:50', '2017/4/6 00:00', dynamic=False, ax=ax, plot_insample=False)
plt.show()
'''

np.save('result/ARIMA.npy', diff_recover_1)