# 导入模块
import numpy as np
import codecs
import os
import DataPreProcess
import matplotlib.pyplot as plt
import TrainModel
import Visualization
import BuildModel

# 数据导入
# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据

total = './Data/total.vocab'
data_11 = './Data/2013-11-fusion.vocab'
data_12 = './Data/2013-12-fusion.vocab'
max_min_path = './Data/loc_max_min.vocab'

# 处理缺失值
data_without_missing_value = DataPreProcess.ProcessMissingValue(data_11, data_12, city_amount=400, judge_num=7)

# 处理异常值
data_without_abnormal_value = DataPreProcess.ProcessAbnormalValue(data_without_missing_value, city_amount=400, judge_week_num=8, judge_day_num=30)

# 归一化数据
total_data = DataPreProcess.DataNormalization(data_without_abnormal_value, max_min_path, city_amount=400)

# 数据保存
DataPreProcess.SavePreProcessData(total_data, total)

# 预测实验
naive_method_predict, naive_method_label = BuildModel.evaluate_naive_method(total_data)

#保存预测结果
HA_result_path = './Data/HA_result.vocab'
label_path = './Data/labels.vocab'
TrainModel.WriteData(HA_result_path, naive_method_predict)

#解归一化
poly_pd = Visualization.DecodeData(HA_result_path, max_min_path)
poly_lb = Visualization.DecodeData(label_path, max_min_path)

# 可视化
from matplotlib.font_manager import _rebuild
_rebuild()

#选取ID=200的栅格区域
id = 200
pd_id = poly_pd[:, id]
lb_id = poly_lb[:, id]

#预测结果可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
if lb_id.shape[0] == pd_id.shape[0]:
    index = np.arange(lb_id.shape[0])
    plt.figure(figsize=(15, 7))
    plt.plot(index, lb_id, c='b', label='真实值')
    plt.plot(index, pd_id, c='r', label='预测值')
    plt.plot(index, pd_id - lb_id, c='k', label='误差值')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
    dt = list(range(len(lb_id)))
    plt.xticks(range(1, len(dt), 24), date, rotation=0)
    # legend设置图例
    plt.legend(loc='best', fontsize=15)
    plt.title("HA 预测值、实际值和误差分布图 ID=200", fontsize=20)
    plt.xlabel('时间', fontsize=15)
    plt.ylabel('流量', fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/5-4-1.svg', format='svg')
    plt.show()

# 模型性能评价
RMSE =Visualization.CalculateRMSE(pd_id.reshape((-1, 1)), lb_id.reshape((-1, 1)))
MAE = Visualization.CalculateMAE(pd_id.reshape((-1, 1)), lb_id.reshape((-1, 1)))
R2 = Visualization.CalculateR2score(pd_id.reshape((-1, 1)), lb_id.reshape((-1, 1)))
print('HA -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (RMSE, MAE, R2))
