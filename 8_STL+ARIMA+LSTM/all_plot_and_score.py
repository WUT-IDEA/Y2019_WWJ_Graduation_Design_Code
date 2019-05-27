import pandas as pd
import matplotlib.pyplot as plt
import  math
import numpy as np

def score_action2(actual,pre):
    actual= np.array(actual)
    purchase_pre = np.array(pre)
    sum = 0
    for i in range(len(actual)):
        # sum += abs(purchase_pre[i] - actual[i])
        sum += (purchase_pre[i] - actual[i])*(purchase_pre[i] - actual[i])
    sum = math.sqrt(sum / 30.0)
    # sum = (sum / 31.0)
    return sum

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

# 组合申购预测值对比图和打分
def plot_total_purchase(title):
    actualTable = pd.read_csv('actual.csv', index_col='timestamp', date_parser=dateparse)
    actualTable = (actualTable.resample('D').mean().interpolate('linear'))
    actual = actualTable['actual']
    pre = []
    for i in range(len(trend)):
        pre.append(trend[i] + seasonal[i] + resid[i])
    print(pre)
    # 画图
    plt.title(title)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(actual, linewidth=1, label='actual')
    plt.plot(index, pre, linewidth=1, label='pre')
    plt.legend(loc='upper right')  # 显示图例，设置图例的位置
    plt.show()
    # print(len(pre))
    print(score_action(pre, actual[:-1]))
    print(score_action2(pre,actual[:-1]))

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式

# 单一分量预测趋势图
def plot_part(preindex,pre,partName):
    trendTable = pd.read_csv('stl_component_7.csv', index_col='timestamp', date_parser=dateparse)
    trendTable = (trendTable.resample('D').mean().interpolate('linear'))
    print(trendTable)
    trainIndex = trendTable.index[:-31]
    print(trainIndex)
    train = trendTable[partName][:-31]
    print(train)
    plt.title('LSTM_'+partName+'_Predict')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(trainIndex,train, linewidth=1, label='train')
    plt.plot(preindex, pre, linewidth=1, label='pre')
    plt.legend(loc='upper right')  # 显示图例，设置图例的位置
    plt.show()

# 预测trend
trendTable = pd.read_csv('LSTM_7_100_trend.csv', index_col='timestamp', date_parser=dateparse)
# trendTable = pd.read_csv('arima_trend_pre.csv', index_col='timestamp', date_parser=dateparse)
trendTable = (trendTable.resample('D').mean().interpolate('linear'))
index = trendTable.index
trend = trendTable['pre']
# 预测seasonal
# seasonalTable = pd.read_csv('arima_seasonal_pre.csv', index_col='timestamp', date_parser=dateparse)
seasonalTable = pd.read_csv('LSTM_7_100_seasonal.csv', index_col='timestamp', date_parser=dateparse)
seasonalTable = (seasonalTable.resample('D').mean().interpolate('linear'))
seasonal = seasonalTable['pre']
# 预测resid
residTable = pd.read_csv('arima_resid_pre.csv', index_col='timestamp', date_parser=dateparse)
# residTable = pd.read_csv('LSTM_7_100_resid.csv', index_col='timestamp', date_parser=dateparse)
residTable = (residTable.resample('D').mean().interpolate('linear'))
resid = residTable['pre']

# plot_part(index,np.array(resid),'resid')
plot_total_purchase('lT_lS_aR')


