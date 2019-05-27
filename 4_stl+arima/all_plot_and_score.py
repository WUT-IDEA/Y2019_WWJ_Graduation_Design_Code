import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def score_action(pre,actual):
    sum_cent=[]
    for i in range(len(actual)):
        sum_cent.append(abs(actual[i]-pre[i])/actual[i])
    sum=0
    for i in range(len(sum_cent)):
        if sum_cent[i] < 0.05:
            sum += 10
        elif sum_cent[i] < 0.1:
            sum += 8
        elif sum_cent[i] < 0.15:
            sum += 6
        elif sum_cent[i] < 0.2:
            sum += 4
        elif sum_cent[i] < 0.25:
            sum += 2
        elif sum_cent[i] < 0.3:
            sum += 1
        else:
            sum += 0
    return sum

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式

# 实际八月数据
actualTable = pd.read_csv('actual.csv', index_col='timestamp', date_parser=dateparse)
actualTable = (actualTable.resample('D').mean().interpolate('linear'))
actual = actualTable['actual']
# 预测trend
trendTable = pd.read_csv('stl_7_trend_pre.csv', index_col='timestamp', date_parser=dateparse)
trendTable = (trendTable.resample('D').mean().interpolate('linear'))
index = trendTable.index
trend = trendTable['trend']
# 预测seasonal
seasonalTable = pd.read_csv('stl_7_seasonal_pre.csv', index_col='timestamp', date_parser=dateparse)
seasonalTable = (seasonalTable.resample('D').mean().interpolate('linear'))
seasonal = seasonalTable['seasonal']
# 预测resid
residTable = pd.read_csv('stl_7_resid_pre.csv', index_col='timestamp', date_parser=dateparse)
residTable = (residTable.resample('D').mean().interpolate('linear'))
resid = residTable['resid']
pre=[]
for i in range(len(trend)):
    pre.append(trend[i]+seasonal[i]+resid[i])
print(pre)
# 画图
plt.title('STL(period=7)+ARIMA(7,1,6)_Predict')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.plot(actual, linewidth=1, label='actual')
plt.plot(index,pre, linewidth=1, label='pre')
plt.legend(loc='upper right')  # 显示图例，设置图例的位置
plt.show()

print(score_action(pre,actual))