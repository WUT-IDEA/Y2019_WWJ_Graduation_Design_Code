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
    pre_table = readData('prophet_forecast.csv')
    pre_train_index = pre_table.index[:-31]
    pre_train_val = pre_table['yhat'][:-31]

    pre_test_index = pre_table.index[-31:]
    pre_test_val = pre_table['yhat'][-31:]

    actual = readData('purchase_table.csv')
    actual_train_index = actual.index[:-31]
    actual_train_val = actual['y_log'][:-31]

    actual_test_index = actual.index[-31:]
    actual_test_val = actual['y_log'][-31:]

    # 训练集对比图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(actual_train_index , actual_train_val , label='actual')
    plt.plot(pre_train_index , pre_train_val , label='pre')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

    # 测试集对比图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # plt.plot(actual_test_index , actual_test_val , label='actual')
    # plt.plot(pre_test_index , pre_test_val , label='pre')
    # plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    # plt.show()