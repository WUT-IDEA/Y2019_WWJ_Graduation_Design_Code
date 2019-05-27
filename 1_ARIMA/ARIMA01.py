# -*- coding: utf-8 -*-
import pandas as pd
# import pyflux as pf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ARIMA模型预测
# 参数：目标值 预测天数,ARIMA参数n,m
def pridictFun(amt,preDay,n,m):
    model = pf.ARIMA(data=train, ar=n, ma=m, target=amt, family=pf.Normal())
    model_x = model.fit("MLE")
    model_x.summary()
    # model.plot_predict(h=31,past_values=9,figsize=(15,5))
    predict_result = model.predict(preDay)  # .values.reshape(31,)
    return  predict_result

def writeResult(purchase_test,purchase_pre,str):
    #字典中的key值即为csv中列名
    timestamp = np.array(purchase_pre.index)
    purchase_pre = np.array(purchase_pre[' purchase '])
    actual = np.array(purchase_test[' purchase '])
    dataframe = pd.DataFrame({'actual':actual,'purchase_pre': purchase_pre,'timestamp':timestamp})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("arima_"+str+".csv",index=False,sep=',')
    print("写入成功")


print('start')

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
table = pd.read_csv('purchase_table.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
table = (table.resample('D').mean().interpolate('linear'))
# 取2014年八月的做测试集
test=table[-31:]
train=table[0:-31]

# 平稳性检测
# 存在单位根，因为存在单位根就是非平稳时间序列
# 判断标准 平稳序列
# ADF Test result同时小于1%、5%、10%
# P-value是否非常接近0.（4位小数基本即可）
# train.plot()
# plt.title('train_data')
# plt.show()
# print('原数据ADF')
# print(ADF(train[' purchase ']))

# 自相关序列
# AR（p）模型：自相关系数拖尾，偏自相关系数p阶截尾。
# MA（q）模型：自相关系数q阶截尾，偏自相关系数拖尾。
# ＡＲＭＡ（ｐ，ｑ）模型：自相关系数拖尾，偏自相关系数拖尾。
# plot_acf(train).show()
# print('原数据白噪声：',acorr_ljungbox(train,lags=1))

# 差分后序列
D_train = train.diff().dropna()
# D_train.plot()
# plt.title('diff_total_purchase')
# plt.show()

# print('差分后的ADF')
# print(ADF(D_train[' purchase ']))
plot_acf(D_train).show()
plot_pacf(D_train).show()
# 白噪声
# p值为第二项， 远小于 0.05
# print('差分后的白噪声：',acorr_ljungbox(D_train,lags=1))

# 购买预测
# purchase_pre1=pridictFun(' purchase ',31,9,9)
# purchase_pre2=pridictFun(' purchase ',31,7,7)
# purchase_pre3=pridictFun(' purchase ',31,6,7)
# purchase_pre4=pridictFun(' purchase ',31,7,6)

# writeResult(test,purchase_pre2,"7_7")
# writeResult(test,purchase_pre3,"6_7")
# writeResult(test,purchase_pre4,"7_6")

# 画图
# plt.title('arima')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# plt.plot(test, linewidth=1, label='actual')
# plt.plot(purchase_pre1, linewidth=1, label='predict_9_9')
# plt.plot(purchase_pre2, linewidth=1, label='predict_7_7')
# plt.plot(purchase_pre3, linewidth=1, label='predict_6_7')
# plt.plot(purchase_pre4, linewidth=1, label='predict_7_6')
#
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()
