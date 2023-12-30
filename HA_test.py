# 导入模块
import random

import numpy as np
import codecs
import copy
import os
import DataPreProcess
import matplotlib.pyplot as plt
import Visualization
import sys

total = './Data/total.vocab'
data_11 = './Data/2013-11-fusion.vocab'
data_12 = './Data/2013-12-fusion.vocab'
max_min_path = './Data/loc_max_min.vocab'

# 数据读取

def read_data(file_data):
    # 读取文件
    with codecs.open(file_data, 'r', 'utf-8') as read:
        value = [w.strip() for w in read.readlines()]
    # 打印value_11的类型，大小
    print(type(value), len(value), type(value[0].split()))
    return value


# # 1.2 处理缺失值(零值)
#
# 处理缺失值
data_without_missing_value = DataPreProcess.ProcessMissingValue(data_11, data_12, city_amount=400, judge_num=7)

# 1.3 处理异常值

# 处理异常值
data_without_abnormal_value = DataPreProcess.ProcessAbnormalValue(data_without_missing_value, city_amount=400, judge_week_num=8, judge_day_num=30)

# 1.4 归一化

# 归一化数据
total_data = DataPreProcess.DataNormalization(data_without_abnormal_value, max_min_path, city_amount=400)

# 1.5 保存可喂入学习模型的数据

# 数据集制作

# 制作能够喂入模型的数据
# total_data 是[60,400,24]的数据形状
def save_data(total_data, total):
    counts_str = []
    for i in range(total_data.shape[0]):
        for j in range(total_data.shape[1]):
            count_str = ''
            for k in range(total_data.shape[2]):
                if(k!=total_data.shape[2]-1):
                    count_str += str(total_data[i][j][k]) + ' '
                else:
                    count_str += str(total_data[i][j][k])
            counts_str.append(count_str)
    if os.path.exists(total):
        print('update ' + total)
        os.remove(total)
    with codecs.open(total, 'a', 'utf-8') as w:
        for i in range(len(counts_str)):
            w.write(counts_str[i] + '\n')

    print(len(counts_str))
save_data(total_data, total)

def poly(data):
    poly_2d = np.zeros(168)
    for i in range(168):
        poly_2d[i] = np.sum(data[i])
    return poly_2d

# 预测实验

def evaluate_naive_method(total_data):
    col = 20 #栅格数据的列数
    row = 20 #栅格数据的行数
    def HA(date,hour):
        counter = 0
        historical_average = 0
        while date>=7:
            date -= 7
            historical_average += total_data[date,:,hour]
            counter += 1
        historical_average /= counter
        return historical_average
    batch_maes = []
    naive_method_predict = [[0] * col*row ] * 7 * 24
    naive_method_label = [[0] * col*row ] * 7 *24
    day_pos = 37 #预测的第一天在数据集中的相对位置
    for day in range(7):  #验证集部分 12月18日-12月25日共7天
        day_pos = 37 #预测的第一天在数据集中的相对位置
        #11月8日是一年中的第312天，12月18日是一年中的第352天,间隔40天
        for hour in range(24):
            preds = HA(day+day_pos,hour)
            labels = total_data[day+day_pos,:,hour]
            naive_method_predict[day * 24 + hour] = preds
            naive_method_label[day * 24 + hour] = labels
            mae = np.mean(np.abs(preds-labels))
            batch_maes.append(mae)
    print(np.mean(batch_maes))
    return np.array(naive_method_predict),np.array(naive_method_label)

def DecodeData(data, max_min_path):
    # 解归一化
    max_min_str = []
    with codecs.open(max_min_path, 'r', 'utf-8') as r:
        max_min_str = [line for line in r.readlines()]
    max_min = []
    for i in range(len(max_min_str)):
        max_min.append([float(value) for value in max_min_str[i].split()])

    # 转置后为2*24000，即每个城市的最大值和最小值
    max_min = np.array(max_min).T
    data = data*(max_min[0]-max_min[1])+max_min[1]
    return data

naive_method_predict,naive_method_label = evaluate_naive_method(total_data)

poly_pd = DecodeData(naive_method_predict, max_min_path)
poly_lb = DecodeData(naive_method_label, max_min_path)

# poly_pd = poly(naive_method_predict)
# poly_lb = poly(naive_method_label)

# 可视化

from matplotlib.font_manager import _rebuild
_rebuild()

# import shutil
# import matplotlib
#
# shutil.rmtree(matplotlib.get_cachedir())


def show_line_chart(result, label):
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['font.serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if label.shape[0] == result.shape[0]:
        index = np.arange(label.shape[0])
        # figsize 设置图形的长和宽，第一个为长，第二个为宽
        plt.figure(figsize=(15,7))
        # g表示green，r表示red
        plt.plot(index, label, c='b', label='真实值')
        plt.plot(index, result, c='r', label='预测值')
        plt.plot(index, result-label, c='k', label='误差值')
        #plt.locator_params(axis = 'x', nbins = 8)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
        dt = list(range(len(label)))
        plt.xticks(range(1,len(dt),24), date, rotation=0)
        # legend设置图例
        plt.legend(loc = 'best',fontsize = 15)
        plt.title("HA 预测值、实际值和误差分布图 ID=200",fontsize=20)
        plt.xlabel('时间',fontsize=15)
        plt.ylabel('流量',fontsize=15)
        plt.tight_layout()
        plt.savefig('./results/5-4-1.svg', format='svg')
        plt.show()

    else:
        print('Wrong Data!')
id = 200
pd_id = poly_pd[:, id].rashaped(-1, 1)
lb_id = poly_lb[:, id].rashaped(-1, 1)
show_line_chart(pd_id,lb_id)

# 模型性能评价


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
RMSE =mean_squared_error(pd_id, lb_id)**0.5
MAE = mean_absolute_error(pd_id, lb_id)
R2 = r2_score(pd_id, lb_id)
print('HA RMSE={}'.format(RMSE))
print('HA MAE={}'.format(MAE))
print('HA R2={}'.format(R2))
