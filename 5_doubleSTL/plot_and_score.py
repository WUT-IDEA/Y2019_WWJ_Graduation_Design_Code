#
# 将直接预测的seasonal以及trend以及stl分解在预测的resid组合预测 评分
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  math
def score_action(actual,pre):
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    # table = pd.read_csv(fileName+'.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
    # table = (table.resample('D').mean().interpolate('linear'))
    # actual=table['actual']
    # purchase_pre=table['purchase_pre']
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

# 读取数据
def readData(filename):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv(str(filename), parse_dates=True, index_col='timestamp', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    return table

def score_action2(actual,pre):
    actual = np.array(actual)
    pre = np.array(pre)
    sum = 0
    for i in range(len(actual)):
        # sum += abs(purchase_pre[i] - actual[i])
        sum += (pre[i] - actual[i])*(pre[i] - actual[i])
    sum = math.sqrt(sum / 31.0)
    return sum

if __name__ == '__main__':
    trend = readData('stl_7_trend_pre.csv')
    seasonal = readData('stl_7_seasonal_pre.csv')
    resid = readData('stl7resid_arima(9,1,9)_pre.csv')
    date_index = trend.index
    trend_arr = np.array(trend)
    seasonal_arr = np.array(seasonal)
    resid_arr = np.array(resid)
    pre=[]
    for i in range(len(resid_arr)):
        pre.append(resid_arr[i][0]+trend_arr[i][0]+seasonal_arr[i][0])
    print(pre)
    actual = readData('purchase_table.csv')[-31:]
    actual_arr = np.array(actual)
    print(actual)

    score = score_action(actual_arr,pre)
    print("1得分：",score)

    score2 = score_action2(actual,pre)
    print("2得分：",score2)

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # plt.title('多次stl分解+arima(9,1,9)')
    # plt.plot(actual, label='actual')
    # plt.plot(date_index,pre, label='pre')
    # plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    # plt.show()