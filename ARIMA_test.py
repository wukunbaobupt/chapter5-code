import csv
import os
import glob
import codecs
import pandas as pd
import re
from datetime import *
# Import libraries
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import statsmodels.api as sm
from sklearn import metrics
from tqdm import tqdm
import DataPreProcess
import statsmodels.api as sm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

# 数据处理

# 1 读取出某区域的流量用于做单点流量预测

# 2 划分训练集和测试集比例为80:20

# df=pd.read_csv(r'base_data.csv','rb')
# #'base_data.csv数据集 168/24=7 天的数据 每一行前有其区域id作为行号
# #示例：读取出区域100的流量用于做单点流量预测
# # print(df.iloc[100][0])
# # print(df.iloc[100,0])
# base_timeseries = [float(x) for x in df.iloc[100][0].split(',')][1:] #通过行号获取数据
# base_timeseries = pd.DataFrame(base_timeseries,columns=['value'])
# #划分训练集和测试集80:20
# train = base_timeseries[0:134]
# chpter5_code = base_timeseries[134:]
# train.head()

# total = './data/total.vocab'
# data_11 = './data/2013-11-fusion.vocab'
# data_12 = './data/2013-12-fusion.vocab'

# def read_data(file_data):
#     # 读取文件
#     with codecs.open(file_data, 'r', 'utf-8') as read:
#         value = [w.strip() for w in read.readlines()]
#     # 打印value_11的类型，大小
#     print(type(value), len(value), type(value[0].split()))
#     return value
# def restore_data(data_str):
#         # 定义一个data变量用来保存数据
#         datas = []
#         for i in range(len(data_str)):
#             # split()主要是将一行数据按空格分开，得到24个数据(表示每个小时的数据)
#             datas.append([float(value) for value in data_str[i].split()])
#         return datas
# #加载数据到all_data
# data_str1 = read_data(data_11)
# data_str2 = read_data(data_12)
# all_data = np.array(restore_data(data_str1) + restore_data(data_str2)) #all_data shape:(24000,24) 400*60*24

# 1.1 数据导入（加载

# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据

total_data_path = './Data/total.vocab'
data_11 =  './data/2013-11-fusion.vocab'
data_12 = './data/2013-12-fusion.vocab'
max_min_path = './Data/loc_max_mix.vocab'

with open(data_11,"r") as f:  #设置文件对象
    print(f.readline().strip())

# # 1.2 处理缺失值(零值)
#
# 处理缺失值
data_without_missing_value = DataPreProcess.ProcessMissingValue(data_11, data_12, city_amount=400, judge_num=7)

# 1.3 处理异常值

# 处理异常值
data_without_abnormal_value = DataPreProcess.ProcessAbnormalValue(data_without_missing_value, city_amount=400, judge_week_num=8, judge_day_num=30)

all_data1=data_without_abnormal_value.reshape(60, 400, 24)
all_data2=data_without_abnormal_value.reshape(400, 60*24)

# all_data1=all_data.reshape(60, 400, 24)
# all_data2=all_data.reshape(400, 60*24)
id = 200
base_timeseries = pd.Series(all_data1[7:44,id,:].reshape(888))
train = base_timeseries[0:720]
test = base_timeseries[720:888]
train.head()
# 模型参数选择 p,d,q

# 1 差分处理，确定d

# 2 画ACF和PACF图，确定p,q
# 确定时间序列的差分
fig = plt.figure(figsize=(20,16))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 原始数据
ax0= fig.add_subplot(311)
train.plot(ax=ax0)
ax0.set_xlabel('时间/小时', fontsize=20)
ax0.set_ylabel('流量', fontsize=20)
ax0.set_title('原始训练数据 ID=%d' %(200), fontsize=20)
ax0.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签的字体大小
ax0.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签的字体大小

# 经过1次/2次差分处理后的数据
ax1= fig.add_subplot(312)
diff1 = train.diff(1)
diff1.plot(ax=ax1)
ax1.set_xlabel('时间/小时', fontsize=20)
ax1.set_ylabel('流量', fontsize=20)
ax1.set_title('一阶差分训练数据 ID=%d' %(200), fontsize=20)
ax1.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签的字体大小
ax1.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签的字体大小


ax2= fig.add_subplot(313)
diff2 = train.diff(2)
diff2.plot(ax=ax2)
ax2.set_xlabel('时间/小时', fontsize=20)
ax2.set_ylabel('流量', fontsize=20)
ax2.set_title('二阶差分训练数据 ID=%d' %(200), fontsize=20)
ax2.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签的字体大小
ax2.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签的字体大小


plt.subplots_adjust(hspace=0.5)
plt.savefig('./results/5-4-2.svg', format='svg')
plt.show()

# 分别画出ACF(自相关)和PACF（偏自相关）图像
fig = plt.figure(figsize=(12,8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
 #ACF(自相关)
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff1.dropna(), lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
# ax1.set_xlabel('unknow', fontsize=20)
# ax1.set_ylabel('unknow', fontsize=20)
ax1.set_title('自相关', fontsize=20)
ax1.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签的字体大小
ax1.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签的字体大小
fig.tight_layout()
 #ACF（偏自相关）
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff1.dropna(), lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
# ax2.set_xlabel('unknow', fontsize=20)
# ax2.set_ylabel('unknow', fontsize=20)
ax2.set_title('偏自相关', fontsize=20)
ax2.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签的字体大小
ax2.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签的字体大小
fig.tight_layout()
plt.savefig('./results/5-4-3.svg', format='svg')
plt.show()

# arima算法 从statsmodels库中调用ARIMA算法实例
# 输入历史数据train_data，预测数据test_data和（p，d，q）顺序的order
# 输出为预测结果predictions_f
def arima_predict(train_data, test_data, order):
    history = train_data
    predictions_f = []
    for t in tqdm(range(len(test_data))):
        model = sm.tsa.ARIMA(history, order=order,
                             # history表示观察到的时间序列过程y。
                             # order表示自回归，差异和移动平均值分量的模型的（p，d，q）顺序。 d始终是整数，而p和q可以是整数或整数列表。
                             )
        model_fit = model.fit()
        yhat_f = model_fit.forecast()[0]
        predictions_f.append(yhat_f)
        history.append(test_data[t])

    return predictions_f

# 绘图设置
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

#这里选用(1,1,1)的（p，d，q）顺序进行实验，实际实验时可选取不同的order对结果进行观测并分析原因。
predict_value = arima_predict(train.values.tolist(),test.values.tolist(),(1,1,1))
predict_value = pd.Series(predict_value,index=test.index)

# error = abs(chpter5_code - predict_value)
# error = pd.Series(error,index=chpter5_code.index)
# visualization
plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(np.arange(len(test.index)), test.values, color='b', label='真实值')

plt.plot(np.arange(len(predict_value.index)), predict_value, color='r', label='预测值')
plt.plot(np.arange(len(predict_value.index)), predict_value - test.values, color='k', label='误差值')
dt = list(range(len(predict_value)))
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
plt.xticks(range(1, len(dt), 24), date, rotation=0)
# plt.plot(chpter5_code,label = "original")
# plt.plot(predict_value,label = "predicted")
# plt.plot(chpter5_code.index,error,label='error')
plt.title("ARIMA 预测值、实际值和误差分布图 ID=200 ", fontsize=20)
plt.xlabel("时间", fontsize=15)
plt.ylabel("流量", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-4-4.svg', format='svg')
plt.show()



RMSE =mean_squared_error(test, predict_value)**0.5
MAE = mean_absolute_error(test, predict_value)
R2 = r2_score(test, predict_value)
print('ARIMA RMSE={}'.format(RMSE))
print('ARIMA MAE={}'.format(MAE))
print('ARIMA R2={}'.format(R2))

