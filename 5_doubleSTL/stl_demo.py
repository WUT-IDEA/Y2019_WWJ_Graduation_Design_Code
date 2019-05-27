import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

# 结果写入csv文件
def writeResult(purchase_test,purchase_pre,file_name):
    #字典中的key值即为csv中列名
    timestamp = np.array(purchase_pre.index)
    purchase_pre = np.array(purchase_pre['drift+seasonal'])
    actual = np.array(purchase_test[' purchase '])
    print(actual)
    print('-'*20)
    print(purchase_pre)
    dataframe = pd.DataFrame({'actual':actual,'purchase_pre': purchase_pre,'timestamp':timestamp})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(file_name+".csv",index=False,sep=',')
    print("写入成功")

# 预测函数 返回八月的实际值 预测值
def getPredictData(name,p_period):
    dateparse=lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')# 时间格式
    table= pd.read_csv(str(name),parse_dates=True,index_col='timestamp',date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    print(table)
    # 训练集   测试集
    train_table = table[:-31]
    test_table = table[-31:]
    # 运用stl模型
    table_stl = decompose(train_table, period=p_period)
    # table_stl.plot()
    # plt.show()
    # 分解
    observed = table_stl.observed
    trend = table_stl.trend
    seasonal = table_stl.seasonal
    resid = table_stl.resid
    fcast = forecast(table_stl, steps=31, fc_func=drift, seasonal=True)
    return  observed,trend,seasonal,resid

if __name__ == '__main__':

    # 获得stl模型预测值和实际至
    observed,trend,seasonal,resid=getPredictData('purchase_table.csv',7)
    resid_stl = decompose(resid, period=7)
    resid_observed = resid_stl.observed
    resid_trend = resid_stl.trend
    resid_seasonal = resid_stl.seasonal
    resid_resid = resid_stl.resid
    resid_stl.plot()
    plt.show()

    timestamp = np.array(resid_observed.index)
    observed = np.array(resid_observed[' purchase '])
    trend = np.array(resid_trend[' purchase '])
    seasonal = np.array(resid_seasonal[' purchase '])
    resid = np.array(resid_resid[' purchase '])
    dataframe = pd.DataFrame(
        {'observed': observed, 'trend': trend, 'seasonal': seasonal, 'resid': resid, 'timestamp': timestamp})
    print(dataframe)
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("stl7resid_component_7.csv", index=False, sep=',')
    print("写入成功")


    # timestamp = np.array(observed.index)
    # observed = np.array(observed[' purchase '])
    # trend = np.array(trend[' purchase '])
    # seasonal = np.array(seasonal[' purchase '])
    # resid = np.array(resid[' purchase '])
    # dataframe = pd.DataFrame({'observed': observed, 'trend': trend,'seasonal':seasonal,'resid':resid,'timestamp': timestamp})
    # print(dataframe)
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("stl_component_7.csv", index=False, sep=',')
    # print("写入成功")

    # writeResult(purchase_test30,purchase_pre30,'STL_30')
    # writeResult(purchase_test35,purchase_pre35,'STL_35')
    # writeResult(purchase_test365,purchase_pre365,'STL_365')

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # # plt.title('period=7')
    # plt.plot(purchase_test7, label='actual')
    # plt.plot(purchase_pre7, label='period=7')
    # plt.plot(purchase_pre30, label='period=30')
    # plt.plot(purchase_pre35, label='period=35')
    # plt.plot(purchase_pre365, label='period=365')

    # plt.legend(loc='upper left')  # 显示图例，设置图例的位置
    # plt.show()




