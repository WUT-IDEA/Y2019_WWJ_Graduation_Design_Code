import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def score_action(fileName):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
    table = pd.read_csv(fileName+'.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    actual=table['actual']
    purchase_pre=table['purchase_pre']
    sum_cent=[]
    for i in range(len(actual)):
        sum_cent.append(abs(actual[i]-purchase_pre[i])/actual[i])
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

print("LSTM_2_100:",score_action("LSTM_2_100"))
print("LSTM_5_100:",score_action("LSTM_5_100"))
print("LSTM_7_100:",score_action("LSTM_7_100"))
print("LSTM_15_100:",score_action("LSTM_15_100"))
print("LSTM_20_100:",score_action("LSTM_20_100"))
print("LSTM_30_100:",score_action("LSTM_30_100"))


