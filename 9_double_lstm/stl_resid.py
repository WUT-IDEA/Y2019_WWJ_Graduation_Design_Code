import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

# 预测函数 返回八月的实际值 预测值
def getPredictData(name,p_period):
    dateparse=lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')# 时间格式
    table= pd.read_csv(str(name),parse_dates=True,index_col='timestamp',date_parser=dateparse)
    table = (table.resample('D').mean().interpolate('linear'))
    print(table)
    # 训练集   测试集
    # train_table = table[:-31]
    # test_table = table[-31:]
    # 运用stl模型
    table_stl = decompose(table, period=p_period)
    table_stl.plot()
    plt.show()
    # 分解
    observed = table_stl.observed
    trend = table_stl.trend
    seasonal = table_stl.seasonal
    resid = table_stl.resid
    # fcast = forecast(table_stl, steps=31, fc_func=drift, seasonal=True)
    return  observed,trend,seasonal,resid

if __name__ == '__main__':

    # 获得stl模型预测值和实际至
    observed,trend,seasonal,resid=getPredictData('stl_7_resid.csv',7)
    print(trend)
    timestamp = np.array(observed.index)
    observed = np.array(observed['resid'])
    trend = np.array(trend['resid'])
    seasonal = np.array(seasonal['resid'])
    resid = np.array(resid['resid'])
    dataframe = pd.DataFrame(
        {'observed': observed, 'trend': trend, 'seasonal': seasonal, 'resid': resid, 'timestamp': timestamp})
    print(dataframe)
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("stl7resid_component_7.csv", index=False, sep=',')
    print("写入成功")



