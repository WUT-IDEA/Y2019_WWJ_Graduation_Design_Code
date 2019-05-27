

# 创建 时间 总购入 总赎回 的表\\


import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
# from date_utils import get_gran, format_timestamp

# 创建 时间 总购入 总赎回 的表
def purchase_and_redem_table(index,X1):
    dataframe = pd.DataFrame({ 'timestamp':index,' purchase ': X1})
    print (dataframe)
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("redeem_table.csv", index=False, sep=',')
    print("写入完成")


dateparse=lambda dates:pd.datetime.strptime(dates,'%Y%m%d')# 时间格式
user_balance= pd.read_csv(r'C:/Users/Administrator/Desktop/demo/ARIMA-1105/user_balance_table.csv',parse_dates=True,index_col='report_date',date_parser=dateparse)
user_balance=user_balance.fillna(0)# 填充空白值
df = user_balance.groupby(by=['report_date']).sum()
print (df)
df_purchase=df['total_purchase_amt']
df_redeem = df['total_redeem_amt']
purchase_and_redem_table(df_purchase.index,df_redeem)


