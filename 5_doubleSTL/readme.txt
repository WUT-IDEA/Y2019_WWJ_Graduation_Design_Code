stl_demo.py
将数据集第一次stl分解获得分量 
写入文件  stl7resid_component_7.csv

arima.py
将  stl第一次分解后的resid的stl分解分量  利用arima(9,1,9)分别预测 
线性组合成 8月resid的预测值

plot_and_score.py
将直接预测的seasonal以及trend以及stl分解在预测的resid组合预测 评分