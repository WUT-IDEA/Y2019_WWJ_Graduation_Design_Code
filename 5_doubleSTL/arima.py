# -*- coding: utf-8 -*-
import pandas as pd
import pyflux as pf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ARIMA模型预测
# 参数：目标值 预测天数,ARIMA参数n,m
def pridictFun(amt,preDay,n,m):
    model = pf.ARIMA(data=dataTable, ar=n, ma=m, target=amt, family=pf.Normal())
    model_x = model.fit("MLE")
    model_x.summary()
    # model.plot_predict(h=31,past_values=9,figsize=(15,5))
    predict_result = model.predict(preDay)  # .values.reshape(31,)
    return  predict_result

def writeResult(purchase_test,purchase_pre):
    #字典中的key值即为csv中列名
    timestamp = np.array(purchase_pre.index)
    purchase_pre = np.array(purchase_pre[' purchase '])
    actual = np.array(purchase_test[' purchase '])
    dataframe = pd.DataFrame({'actual':actual,'purchase_pre': purchase_pre,'timestamp':timestamp})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("arima_9_9.csv",index=False,sep=',')
    print("写入成功")

def plot_component(tit,actual,predata):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(tit)
    plt.plot(indexTrain,actual, linewidth=1, label='train')
    plt.plot(indexTest,predata, linewidth=1, label='pre')
    plt.legend(loc='upper right')  # 显示图例，设置图例的位置
    plt.show()


# 读取第一次stl分解获得的resid以及分解分量
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
dataTable = pd.read_csv('stl7resid_component_7.csv', index_col='timestamp', date_parser=dateparse)
dataTable = (dataTable.resample('D').mean().interpolate('linear'))

observed = dataTable['observed']
trend = dataTable['trend']
seasonal = dataTable['seasonal']
resid = dataTable['resid']

# 第一次获得的resid分解分量图
# resid.plot()
# plt.show()

# trend归一化
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0,1))
trend = np.array(dataTable['trend']).reshape(-1,1)
trend_reshape=scaler.fit_transform(trend)

# 预测分量
trend_pre=pridictFun('trend_reshape',31,9,9)
trend_pre = scaler.inverse_transform(trend_pre)
seasonal_pre=pridictFun('seasonal',31,9,9)
resid_pre=pridictFun('resid',31,9,9)

# 画RESID分量预测图
indexTrain=np.array(range(396))
indexTest = np.array(range(396,427))
# plot_component('RESID_trend_predict',trend,trend_pre)
# plot_component('RESID_seasonal_predict',seasonal,seasonal_pre)
# plot_component('RESID_resid_predict',resid,resid_pre)

# 分量线性组合
seasonal_pre=np.array(seasonal_pre['seasonal'])
resid_pre=np.array(resid_pre['resid'])
# 预测的resid线性组合值
pre=[]
for i in range(len(trend_pre)):
    pre.append(trend_pre[i][0]+seasonal_pre[i]+resid_pre[i])
print("预测的八月resid:",pre)
print("训练集的resid:",observed)

# 画RESID包括八月预测的趋势图
observed=np.array(observed)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.plot(indexTrain,observed, linewidth=1, label='seasonal_train')
plt.plot(indexTest,pre, linewidth=1, label='seasonal_predict')
plt.legend(loc='upper right')  # 显示图例，设置图例的位置
plt.show()

dataframe = pd.DataFrame({'resid_pre': pre, 'timestamp': indexTest})
print(dataframe)
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("stl7resid_arima(9,1,9)_pre.csv", index=False, sep=',')
print("写入成功")