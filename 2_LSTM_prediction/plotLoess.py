import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd

table = pd.read_csv('lstmloess.csv')['loess']
table = np.array(table)
print(table)
arr = []
for i in table:
    arr.append(float(i.split(' ')[-1]))

plt.title('loess')
plt.plot(range(0,len(arr)),arr, label='predict')
# plt.plot(index,testY,  label='actual')
# plt.legend(loc='best')
# plt.title('LSTM purchase prediction')
plt.show()