import numpy as np
import codecs
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import DataPreProcess
import BuildModel
import Visualization
import tensorflow
import keras
import math

# 数据导入
# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据
total = './data/total.vocab'
data_11 = './data/2013-11-fusion.vocab'
data_12 = './data/2013-12-fusion.vocab'

def read_data(file_data):
    # 读取文件
    with codecs.open(file_data, 'r', 'utf-8') as read:
        value = [w.strip() for w in read.readlines()]
    # 打印value_11的类型，大小
    print(type(value), len(value), type(value[0].split()))
    return value

def restore_data(data_str):
        # 定义一个data变量用来保存数据
        datas = []
        for i in range(len(data_str)):
            # split()主要是将一行数据按空格分开，得到24个数据(表示每个小时的数据)
            datas.append([float(value) for value in data_str[i].split()])
        return datas

#加载数据到all_data
data_str1 = read_data(data_11)
data_str2 = read_data(data_12)
all_data = np.concatenate(restore_data(data_str1) + restore_data(data_str2)) #all_data shape:(24000,24) 400*60*24
all_data1=all_data.reshape(60, 400, 24)
all_data2=all_data.reshape(400, 60*24)

# 查看某地区某天24小时的连续时序流量变化

loc_id_1 = 100#地区id（0-399）
loc_id_2 = 200#地区id（0-399）

day=12 #两个月中的第几天
index=np.arange(all_data1.shape[2])
label = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
plt.bar(index, np.array(all_data1[day,loc_id_1,:]))
plt.legend(loc = 'best')
plt.title("11月12日的流量变化 ID=%d" %(loc_id_1), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.xticks(np.arange(24), labels=label, rotation=-90, fontsize=12)
plt.ylim(0,math.ceil(max(np.concatenate((np.array(all_data1[day,loc_id_1,:]), np.array(all_data1[day,loc_id_2,:])))) / 50) * 50)
plt.subplot(1,2,2)
plt.bar(index, np.array(all_data1[day,loc_id_2,:]))
plt.legend(loc = 'best')
plt.title("11月12日的流量变化 ID=%d" %(loc_id_2), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.xticks(np.arange(24), labels=label, rotation=-90, fontsize=12)
plt.ylim(0,math.ceil(max(np.concatenate((np.array(all_data1[day,loc_id_1,:]), np.array(all_data1[day,loc_id_2,:])))) / 50) * 50)
plt.tight_layout()
plt.savefig('./results/5-2-3.svg', format='svg')
plt.show()

# 查看某地区7天连续的时序流量变化

loc_id_1 = 100#地区id（0-399）
loc_id_2 = 200#地区id（0-399）
N_day = int(all_data2.shape[1]*7/60)
index=np.arange(N_day)
plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
plt.plot(index, np.array(all_data2[loc_id_1, 0:N_day]))
plt.legend(loc = 'best')
plt.title("7天内的流量变化 ID=%d" %(loc_id_1), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.ylim(0,math.ceil(max(np.concatenate((np.array(all_data2[loc_id_1, 0:N_day]), np.array(all_data2[loc_id_2, 0:N_day])))) / 50) * 50)
date = ['星期六', '星期日', '星期一', '星期二', '星期三', '星期四', '星期五']
dt = list(range(len(index)))
plt.xticks(range(1, len(dt), 24), date, rotation=0, fontsize=15)
plt.subplot(1,2,2)
plt.plot(index, np.array(all_data2[loc_id_2, 0:N_day]))
plt.legend(loc = 'best')
plt.title("7天内的流量变化 ID=%d" %(loc_id_2), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.ylim(0,math.ceil(max(np.concatenate((np.array(all_data2[loc_id_1, 0:N_day]), np.array(all_data2[loc_id_2, 0:N_day])))) / 50) * 50)
date = ['星期六', '星期日', '星期一', '星期二', '星期三', '星期四', '星期五']
dt = list(range(len(index)))
plt.xticks(range(1, len(dt), 24), date, rotation=0, fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-2-4.svg', format='svg')
plt.show()

# 查看某地区30天连续的时序流量变化

loc_id = 100#地区id（0-399）
N_day = int(all_data2.shape[1])
index=np.arange(N_day)
plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(index, np.array(all_data2[loc_id, 0:N_day]),)
plt.legend(loc = 'best')
plt.title("60天内的流量变化 ID=%d" %(loc_id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
date = ['11月1日', '11月8日', '11月15日', '11月22日', '11月29日', '12月6日', '12月13日', '12月20日','12月27日']
dt = list(range(len(all_data2[loc_id, 0:N_day])))
plt.xticks(range(1, len(dt), 168), date, rotation=0, fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-2-5.svg', format='svg')
plt.show()


#查看城市中某天的空间流量分布情况

hour_1=15
hour_2=21
day=16 #两个月中的第几天

fig = plt.figure(figsize=(15,5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

vmax= np.max(np.concatenate((all_data1.reshape(60,20,20,24)[day,:,:,hour_1].flatten(), all_data1.reshape(60,20,20,24)[day,:,:,hour_2].flatten())))
vmin= np.min(np.concatenate((all_data1.reshape(60,20,20,24)[day,:,:,hour_1].flatten(), all_data1.reshape(60,20,20,24)[day,:,:,hour_2].flatten())))

plt.subplot(121)
ax1 =sns.heatmap(all_data1.reshape(60,20,20,24)[day,:,:,hour_1], square=True, vmin=vmin, vmax=vmax)
ax1.set_title("11月16日下午3点的真实流量" , fontsize=20)
plt.xticks(np.arange(0.5, 20.5, 1), labels=np.arange(1, 21, 1))  # 设置 x 轴刻度
plt.yticks(np.arange(0.5, 20.5, 1), labels=np.arange(1, 21, 1))  # 设置 y 轴刻度
plt.xlim()

plt.subplot(122)
ax1 =sns.heatmap(all_data1.reshape(60,20,20,24)[day,:,:,hour_2], square=True, vmin=vmin, vmax=vmax)
ax1.set_title("11月16日晚上9点的真实流量" , fontsize=20)
plt.xticks(np.arange(0.5, 20.5, 1), labels=np.arange(1, 21, 1))  # 设置 x 轴刻度
plt.yticks(np.arange(0.5, 20.5, 1), labels=np.arange(1, 21, 1))  # 设置 y 轴刻度

plt.tight_layout()
plt.savefig('./results/5-2-6.svg', format='svg')
plt.show()
