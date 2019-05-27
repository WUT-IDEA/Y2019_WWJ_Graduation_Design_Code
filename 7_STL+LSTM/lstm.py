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

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def writeResult(actual,purchase_pre,index,filename):
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'actual':actual,'pre': purchase_pre,'timestamp':index})
    print(dataframe)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filename+".csv",index=False,sep=',')
    print("写入成功")


# 获取每日购买数据
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  # 时间格式
table = pd.read_csv('stl_component_7.csv', parse_dates=True, index_col='timestamp', date_parser=dateparse)
table = (table.resample('D').mean().interpolate('linear'))
index=table.index

# table['seasonal'].plot()
# plt.show()

trainData = np.array(table['resid'])
trainData_reshape = trainData.reshape(-1,1)
print(trainData)
# 归一化
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
purchase_reshape = scaler.fit_transform(trainData_reshape)
print(purchase_reshape)


# 训练集 测试集切分
look_back = 7
train, test = purchase_reshape[0:-31], purchase_reshape[-31-look_back:]

# 输入格式
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX=trainX[:-look_back+1]
trainY=trainY[:-look_back+1]

# 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# 模型构造
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model_prob = model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=2)

#预测
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict).reshape(1,-1)[0]
testY = scaler.inverse_transform([testY])[0]

# 绘制八月对比图
# print('testY')
# print(testY)
# print('testPredict')
testPredict = testPredict.tolist()[1:]
testPredict.append(testY[len(testY)-1])
# print(testPredict)

index=index[-31:-1]
plt.plot(index,testPredict, label='predict')
plt.plot(index,testY,  label='actual')
plt.legend(loc='best')
plt.title('LSTM trend prediction')
plt.show()

writeResult(testY,testPredict,index,'LSTM_7_100_resid')

