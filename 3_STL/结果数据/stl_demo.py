import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
# 画图
def drawPlot(test,title):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.plot(test, color="blue", label=title)
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

dataset = sm.datasets.co2.load_pandas()
obs = dataset.data
obs = (obs.resample('D').mean().interpolate('linear'))

short_obs = obs.head(10000)
short_stl = decompose(short_obs, period=365)
stl_o = short_stl.observed
stl_t = short_stl.trend
stl_s = short_stl.seasonal
stl_r = short_stl.resid
drawPlot(stl_o,'observed')
drawPlot(stl_t,'trend')
drawPlot(stl_s,'seasonal')
drawPlot(stl_r,'resid')


# fcast = forecast(short_stl, steps=8000, fc_func=drift)
fcast = forecast(short_stl, steps=8000, fc_func=drift, seasonal=True)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.plot(obs, '--', label='实际值')
plt.plot(short_obs, '--', label='训练集')
plt.plot(short_stl.trend, ':', label='训练集趋势')
plt.plot(fcast, '-', label='预测值')

plt.xlim('1970','2004');
plt.ylim(330,380);
plt.legend()
plt.show()

