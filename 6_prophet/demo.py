import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('purchase_table.csv')
df['y'] = np.log(df['y'])

# 划分训练集测试机
train = df[:-31]
test = df[-31:]

# 画图
# plt.title('train')
# plt.plot(train['ds'],train['y'],linewidth=1, label='actual')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()

# plt.title('test')
# plt.plot(test['ds'],test['y'],linewidth=1, label='actual')
# plt.legend(loc='upper right')  # 显示图例，设置图例的位置
# plt.show()

# 训练模型
m = Prophet()
m.fit(train)

# 预测日期格式处理
future = m.make_future_dataframe(periods=31)
print(future)
forecast = m.predict(future)
forecast.to_csv("prophet_forecast.csv", index=False, sep=',')
print("写入成功")