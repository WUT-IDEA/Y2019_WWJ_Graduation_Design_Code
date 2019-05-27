import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
# 画时序分解的图
def draw_STL_info(title,o,t,s,i):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(title)
    plt.plot(o, color="blue", label='observed')
    plt.plot(t, color='red', label='trend')
    plt.plot(s, color='green', label='seasonal')
    plt.plot(i, color='black', label='remainder')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

# 结果写入csv文件
def writeResult(redeem_pre,purchase_pre):
    datetime=range(20140901,20140931)
    #字典中的key值即为csv中列名
    redeem_pre = np.array(redeem_pre['drift+seasonal'])
    purchase_pre = np.array(purchase_pre['drift+seasonal'])
    dataframe = pd.DataFrame({'report_date':datetime,'purchase': purchase_pre,'redeem' : redeem_pre })
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("tc_comp_predict_table_35.csv",index=False,sep=',')
    print("写入成功")

# 预测函数 返回八月的实际值 预测值
def getPredictData(name):
    dateparse=lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')# 时间格式
    table= pd.read_csv(str(name),parse_dates=True,index_col='timestamp',date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    # 训练集
    train_table = table[274:-31]
    # 测试集
    test_table = table[-31:]
    # 运用stl模型
    table_stl1 = decompose(train_table, period=7)
    o = table_stl1.observed
    t = table_stl1.trend
    s = table_stl1.seasonal
    i = table_stl1.resid
    # print(i)

    draw_STL_info('第一次STL时序分解',o, t, s,i)
    fcast = forecast(table_stl1, steps=30, fc_func=drift, seasonal=True)
    return  test_table,fcast

# 画图
def drawPlot(title,test,pre):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(title)
    plt.plot(test, color="blue", label='实际值')
    plt.plot(pre, color='red', label='预测数据')
    plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    plt.show()

if __name__ == '__main__':
    purchase_test, purchase_pre = getPredictData('purchase_table.csv')
    # redeem_test, redeem_pre = getPredictData('redeem_table.csv')

    # drawPlot("八月买入额情况预测对比", purchase_test, purchase_pre)




