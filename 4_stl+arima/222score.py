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

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式

# 实际八月数据
actualTable = pd.read_csv('actual.csv', index_col='timestamp', date_parser=dateparse)
actualTable = (actualTable.resample('D').mean().interpolate('linear'))
actual = np.array(actualTable['actual'])
# 预测trend
trendTable = pd.read_csv('stl_7_trend_pre.csv', index_col='timestamp', date_parser=dateparse)
trendTable = (trendTable.resample('D').mean().interpolate('linear'))
index = trendTable.index
trend = np.array(trendTable['trend'])
# 预测seasonal
seasonalTable = pd.read_csv('stl_7_seasonal_pre.csv', index_col='timestamp', date_parser=dateparse)
seasonalTable = (seasonalTable.resample('D').mean().interpolate('linear'))
seasonal = np.array(seasonalTable['seasonal'])
# 预测resid
residTable = pd.read_csv('stl_7_resid_pre.csv', index_col='timestamp', date_parser=dateparse)
residTable = (residTable.resample('D').mean().interpolate('linear'))
resid = np.array(residTable['resid'])
pre=[]
for i in range(len(trend)):
    pre.append(trend[i]+seasonal[i]+resid[i])
print(pre)
plot(pre, actual)

sum = 0
for i in range(len(actual)):
    # sum += abs(purchase_pre[i] - actual[i])
    sum += (pre[i] - actual[i]) * (pre[i] - actual[i])
sum = math.sqrt(sum / 31.0)
print(sum)




