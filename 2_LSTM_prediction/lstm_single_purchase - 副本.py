'''
赎回的lstm
'''
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras.models
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential

import pandas as pd
from sklearn.externals import joblib
import os
def err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# # load the dataset
# dataframe = read_csv('./file/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# plt.plot(dataset)
# plt.show()

def writeResult(actual,purchase_pre,index,filename):
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'actual':actual,'purchase_pre': purchase_pre,'timestamp':index})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filename+".csv",index=False,sep=',')
    print("写入成功")

# 获取每日购买数据
dataframe = read_csv('../file/group_by_date.csv', usecols=[3], engine='python')
df = read_csv('../file/group_by_date.csv', index_col='report_date', parse_dates=[0])
dataset = dataframe.values
dataset = dataset.astype('float64')

# 获取每日购买数据
total_purchase_amt_ts = df['total_purchase_amt']

# 归一化
# fix random seed for reproducibility
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print('guiyihua')
print(dataset)
# 归一化后的 训练集 测试集切分
train, test = dataset[0:389], dataset[389:]


# use this function to prepare the train and test datasets for modeling
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# 模型构造
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print('model.summary:',model.summary())
myfile = os.path.exists("lstm.model")
if myfile:
    print("ssss")
else:
    model_prob = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    index=df.index[-31:]
    writeResult(testY,testPredict,index,'LSTM_demo')
# 均方误差计算
errs = err(testY[0],testPredict[:,0])
print("err:",errs)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# 输出到csv
# trainData = pd.DataFrame({'trainpredict':trainPredict[:,0],'actual':trainY[0]})
# testData = pd.DataFrame({'testpredict':testPredict[:,0],'actual':testY[0]})
# trainData.to_csv("train_purchase.csv")
# testData.to_csv("test_purchase.csv")

# 绘制八月对比图
print('testPredict[:,0]:')
print(testPredict[:,0])
print('testY[0]')
print(testY[0])
t=np.array(range(30))
print(t)
plt.plot(t,testPredict[:,0], label='predict')
plt.plot(t,testY[0],  label='actual')
plt.legend(loc='best')
plt.title('LSTM purchase prediction 8')
plt.show()

# shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# date_trainPredictPlot = pd.DataFrame(data=trainPredictPlot, index=total_purchase_amt_ts.index, columns=['value'])
# date_testPredictPlot = pd.DataFrame(data=testPredictPlot, index=total_purchase_amt_ts.index, columns=['value'])
# plt.plot(total_purchase_amt_ts, label='original')
# plt.plot(date_testPredictPlot, label='test')
# plt.plot(date_trainPredictPlot,  label='train')
# plt.legend(loc='best')
# plt.title('LSTM purchase prediction')
# plt.show()
