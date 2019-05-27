import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot(pre,actual):
    plt.title('arima')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(actual, linewidth=1, label='actual')
    plt.plot(pre, linewidth=1, label='pre')
    plt.legend(loc='upper right')  # 显示图例，设置图例的位置
    plt.show()

def score_action(fileName):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv(fileName+'.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    actual= np.array(table['actual'])
    purchase_pre = np.array(table['purchase_pre'])
    plot(purchase_pre,actual)

    sum = 0
    for i in range(len(actual)):
        # sum += abs(purchase_pre[i] - actual[i])
        sum += (purchase_pre[i] - actual[i])*(purchase_pre[i] - actual[i])
    sum = math.sqrt(sum / 31.0)
    # sum = (sum / 31.0)
    return sum

print("STL_7",score_action("STL_7"))
print("STL_30",score_action("STL_30"))
print("STL_35",score_action("STL_35"))
print("STL_365",score_action("STL_365"))


