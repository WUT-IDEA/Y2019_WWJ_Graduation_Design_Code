# 画resid趋势图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def readData(filename):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv(str(filename), parse_dates=True, index_col='timestamp', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    return table

if __name__ == '__main__':
    actual_resid = readData('stl7resid_component_7.csv')
    pre_resid = readData('stl7resid_arima(9,1,9)_pre.csv')
    # print(actual_resid['observed'])
    # print(pre_resid)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title('resid再次STL分解预测后趋势图')
    plt.plot(actual_resid["observed"], label='actual')
    plt.plot(pre_resid, label='pre')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()