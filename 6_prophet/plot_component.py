# 画resid趋势图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def readData(filename):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv(str(filename), parse_dates=True, index_col='ds', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    return table

if __name__ == '__main__':
    table = readData('prophet_forecast.csv')
    index = table.index
    trend = table['trend']
    weekly = table['weekly']
    additive = table['additive_terms']
    multiplicative = table['multiplicative_terms']
    yhat = table['yhat']

    # 测试集对比图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.figure()
    plt.subplot(511)
    plt.plot(index , trend , label='trend')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.subplot(512)
    plt.plot(index , weekly , label='weekly')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.subplot(513)
    plt.plot(index, additive, label='additive')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.subplot(514)
    plt.plot(index, multiplicative, label='multiplicative')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.subplot(515)
    plt.plot(index, yhat, label='yhat')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()