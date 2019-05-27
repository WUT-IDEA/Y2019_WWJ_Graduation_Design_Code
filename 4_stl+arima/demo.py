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

np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0,1))

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
dataTable = pd.read_csv('stl_component_7.csv', index_col='timestamp', date_parser=dateparse)
dataTable = (dataTable.resample('D').mean().interpolate('linear'))

observed = dataTable['observed']
trend = np.array(dataTable['trend']).reshape(-1,1)
# trend_reshape = dataTable['trend_reshape']

trend_reshape=scaler.fit_transform(trend)

seasonal = dataTable['seasonal']
resid = dataTable['resid']



# 购买预测
trend_pre=pridictFun('trend_reshape',31,9,9)
# seasonal_pre=pridictFun('seasonal',31,9,9)
# resid_pre=pridictFun('resid',31,9,9)

trend_pre = scaler.inverse_transform(trend_pre)


# print(np.array(range(396)))
# print(np.array(range(396,427)))
# print(len(trend))




# 画图
# plt.title('trend')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# plt.title('August_trend_predict')
# plt.plot(trend_reshape, linewidth=1, label='seasonal_train')
# plt.plot(trend_pre, linewidth=1, label='seasonal_predict')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()


# plt.title('seasonal')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# plt.title('August_seasonal_predict')
# plt.plot(seasonal, linewidth=1, label='seasonal_train')
# plt.plot(seasonal_pre, linewidth=1, label='seasonal_predict')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()


# plt.title('resid')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# plt.plot(resid, linewidth=1, label='seasonal_train')
# plt.plot(resid_pre, linewidth=1, label='seasonal_predict')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()