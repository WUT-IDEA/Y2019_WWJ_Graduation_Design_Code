# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
dataTable = pd.read_csv('stl_component_7.csv', index_col='timestamp', date_parser=dateparse)
dataTable = (dataTable.resample('D').mean().interpolate('linear'))
data_train= dataTable['resid']

dataTest = pd.read_csv('stl_7_resid_pre.csv', index_col='timestamp', date_parser=dateparse)
dataTest = (dataTest.resample('D').mean().interpolate('linear'))

# 画图
plt.title('STL_Resid_predict')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.plot(dataTest, linewidth=1, label='resid_predict')
plt.plot(data_train, linewidth=1, label='resid_actual')
plt.legend(loc='upper right')  # 显示图例，设置图例的位置
plt.show()
# index,actual,pre_6_7 = plot_action('6','7')
# index1,actual1,pre_7_6 = plot_action('7','6')
# index2,actual2,pre_7_7 = plot_action('7','7')
# index3,actual3,pre_9_9 = plot_action('9','9')
# # 画图
# plt.title('Total_purchase Predict')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# plt.plot(index, actual, linewidth=1, label='actual')
# plt.plot(index, pre_6_7, linewidth=1, label='arima(6,1,7)')
# plt.plot(index, pre_7_6, linewidth=1, label='arima(7,1,6)')
# plt.plot(index, pre_7_7, linewidth=1, label='arima(7,1,7)')
# plt.plot(index, pre_9_9, linewidth=1, label='arima(9,1,9)')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()