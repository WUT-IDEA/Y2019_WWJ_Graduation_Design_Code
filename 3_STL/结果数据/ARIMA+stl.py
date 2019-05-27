# -*- coding: utf-8 -*-
import pandas as pd
import pyflux as pf
#导入图表库
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
# ARIMA模型预测
# 参数：目标值 预测天数,ARIMA参数n,m
def pridictFun(train_data,amt,preDay,n,m):
    print train_data
    model = pf.ARIMA(data=train_data, ar=n, ma=m, target=amt, family=pf.Normal())
    model_x = model.fit("MLE")
    model_x.summary()
    predict_result = model.predict(preDay)  # .values.reshape(31,)
    return  predict_result

# 把数据写入csv
def writeResult(redeem_pre,purchase_pre):
    datetime=range(20140901,20140931)
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'redeem' : redeem_pre,' purchase ': purchase_pre ,'report_date':datetime  })
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("tc_comp_predict_table.csv",index=False,sep=',')

# 运用一次stl
def stl_once(train_table):
    # 运用stl模型
    table_stl = decompose(train_table, period=35)
    fcast = forecast(table_stl, steps=30, fc_func=drift, seasonal=True)
    plt.title("stl")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(test_table[' purchase '], linewidth=1, label='八月购买实际值')
    plt.plot(fcast['drift+seasonal'], linewidth=1, label='八月购买预测值')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

if __name__ == '__main__':
    # 获取数据
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv('purchase_table.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))

    plt.title("totle_purchase")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(table, linewidth=1)
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

    # 训练集
    train_table = table[:-31]
    # 测试集
    test_table = table[-31:]
    # 运用stl模型
    table_stl = decompose(train_table, period=35)
    # 第一次stl分解
    observed = table_stl.observed
    trend = table_stl.trend
    seasonal = table_stl.seasonal
    resid = table_stl.resid

    fcast = forecast(table_stl, steps=30, fc_func=drift, seasonal=True)
    plt.title("stl")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # plt.plot(trend, linewidth=1, label='趋势')
    plt.plot(fcast['drift+seasonal'], linewidth=1, label='八月购买预测值')
    # plt.plot(table, linewidth=1, label='purchase')
    # plt.plot(table1, linewidth=1, label='redeem')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()
    # 第二次stl以及分解
    # table_stl_2 = decompose(trend, period=365)
    # seasonal_2 = table_stl_2.seasonal
    # resid_2 = table_stl_2.resid
    # trend_2=table_stl_2.trend
    # 购买预测
    # purchase_seasonal_1=pridictFun(seasonal,' purchase ',31,9,9)
    # purchase_trend_1=pridictFun(trend,' purchase ',31,9,9)
    # purchase_resid_1=pridictFun(resid,' purchase ',31,9,9)
    #
    # purchase_trend_2=pridictFun(trend_2,' purchase ',31,9,9)
    # purchase_seasonal_2=pridictFun(seasonal_2,' purchase ',31,9,9)
    # purchase_resid_2=pridictFun(resid_2,' purchase ',31,9,9)

    # purchase_seasonal_1_arr=np.array(purchase_seasonal_1[' purchase '])
    # purchase_trend_1_arr=np.array(purchase_trend_1[' purchase '])
    # purchase_resid_1_arr=np.array(purchase_resid_1[' purchase '])
    #
    # purchase_trend_2_arr = np.array(purchase_trend_2[' purchase '])
    # purchase_seasonal_2_arr = np.array(purchase_seasonal_2[' purchase '])
    # purchase_resid_2_arr=np.array(purchase_resid_2[' purchase '])

    # len=purchase_resid_2_arr.__len__()
    # predict_result=[]
    # for i in range(len):
    #     predict_result.append(purchase_seasonal_1_arr[i]+purchase_trend_1_arr[i]+purchase_resid_1_arr[i])
    # print '预测结果：'
    # print predict_result
    # print test_table
    # 画图
