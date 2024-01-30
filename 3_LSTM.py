import tensorflow as tf
import numpy as np
import warnings
import DataPreProcess
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os

import Visualization
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU,实验结果可复习。
warnings.filterwarnings("ignore")

# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据

total_data_path = './Data/total.vocab'
data_11 = './data/2013-11-fusion.vocab'
data_12 = './data/2013-12-fusion.vocab'
max_min_path = './Data/loc_max_mix.vocab'

with open(data_11,"r") as f:  #设置文件对象
    print(f.readline().strip())

# 处理缺失值
data_without_missing_value = DataPreProcess.ProcessMissingValue(data_11, data_12, city_amount=400, judge_num=7)

# 处理异常值
data_without_abnormal_value = DataPreProcess.ProcessAbnormalValue(data_without_missing_value, city_amount=400, judge_week_num=8, judge_day_num=30)


all_data1 = data_without_abnormal_value.reshape(60, 400, 24)

#选取ID=200的栅格区域
id = 200
#选取第7天-第43天的数据
series =all_data1[7:44,id,:].reshape(888)
time = np.arange(len(series))

#可视化30天内的流量数据
plt.figure(figsize=(15, 7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(time[0:720], series[0:720], '-')
plt.xlabel("时间",fontsize=15)
plt.ylabel("流量",fontsize=15)
plt.title("30天内流量变化 ID=%d" % (id),fontsize=20)
date = ['11月8日', '11月14日', '11月22日', '11月29日', '12月6日']
dt = list(range(len(series[0:720])))
plt.xticks(range(1, len(dt), 7*24), date, rotation=0,fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/5-4-8.svg', format='svg')
plt.show()

# 划分训练集和测试集
t_time = 720
split_time = 600  # 可以尝试600 300
time_train = time[t_time - split_time:t_time]
x_train = series[t_time - split_time:t_time]
time_test = time[t_time:]
X_test = series[t_time:]

window_size = 100  # 可以尝试4 12 24 100
batch_size = 32  # 2^n
if split_time % 3 == 0:
    shuffle_buffer_size = np.float64(split_time / 3)
else:
    shuffle_buffer_size = np.float64(split_time / 2)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # 函数功能：制作基于滑动窗口的数据集，将数据集按照窗口大小和步长进行切分，并
    #          将所有窗口数据展平成一个大的数据集，对数据集随机采样，最后将每个
    #          样本划分为训练数据和目标数据。
    #    输入：series—输入训练数据
    #          window_size—滑动窗口的大小，即每个样本包含多少个连续时间步的数据
    #          batch_size—数据批量大小，即每次训练模型时输入的样本数量
    #          shuffle_buffer—做随机采样使用的缓冲大小，用来打乱数据集中数据顺序
    #    输出：ds.batch(batch_size).prefetch(1) —大小为batch_size的数据集

    series = tf.expand_dims(series, axis=-1)  # 输入训练数据进行一维展平
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)  # 将数据集按照窗口大小和步长进行切分
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)  # 对数据集进行随机采样，以防止训练过程中的过拟合
    ds = ds.map(lambda w: (w[:-1], w[1:]))  # 将每个样本划分为训练数据和目标数据。
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    # 输入训练完成后的模型，和历史序列以及滑动窗口大小。并返回预测结果
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


tf.random.set_seed(30)
np.random.seed(30)

# 得到经过时间窗切片的滑动窗口训练数据的测试训练数据
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)
test_set = windowed_dataset(X_test, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)

# 定义模型
model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv1D(filters=60, kernel_size=10,
    #                        strides=1, padding="causal",
    #                        activation="relu",
    #                        input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(lr=0.001)#优化器 Adam学习率1e-3 （可调节）

#使用上文定义的基于LSTM的model结构，loss函数选为Huber损失，即平滑的平均绝对误差
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
#模型训练 训练伦次epochs (可调节)
history = model.fit(train_set, epochs=100)
model.summary()
# plot_model(model, to_file='results/5-4-6.png', show_shapes=True, show_layer_names=True)

# Loss可视化
loss = history.history['loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'r', label='训练损失')
plt.title('LSTM')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('./results/5-4-9.svg', format='svg')
plt.show()

# 使用进行模型预测
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[t_time - window_size:-1, -1, 0]

# 预测结果可视化
plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(np.arange(len(time_test)), series[time_test], color='b', label='真实值',)
plt.plot(np.arange(len(time_test)), rnn_forecast, color='r', label='预测值',)
plt.plot(np.arange(len(time_test)), rnn_forecast - series[time_test], color='k', label='误差值')
dt = list(range(len(time_test)))
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
plt.xticks(range(1, len(dt), 24), date, rotation=0,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='best',fontsize=15)
plt.title('LSTM 预测值、实际值和误差分布图 ID=%d'%(id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-4-10.svg', format='svg')
plt.show()

#性能结果
RMSE =Visualization.CalculateRMSE(rnn_forecast.reshape((-1, 1)), X_test.reshape((-1, 1)))
MAE = Visualization.CalculateMAE(rnn_forecast.reshape((-1, 1)), X_test.reshape((-1, 1)))
R2 = Visualization.CalculateR2score(rnn_forecast.reshape((-1, 1)), X_test.reshape((-1, 1)))
print('LSTM -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (RMSE, MAE, R2))
