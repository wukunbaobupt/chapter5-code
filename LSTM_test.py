import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import DataPreProcess
import codecs #读取文件内容时，会自动转换为内部的Unicode
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")


# total = './data/total.vocab'
# data_11 = './data/2013-11-fusion.vocab'
# data_12 = './data/2013-12-fusion.vocab'
#
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


# 一、数据预处理

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

id = 200
series =all_data1[7:44,id,:].reshape(888)
time = np.arange(len(series))
plt.figure(figsize=(15, 7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(time[0:720], series[0:720], '-')

plt.xlabel("时间",fontsize=15)
plt.ylabel("流量",fontsize=15)
plt.title("30天内流量变化 ID=%d" % (id),fontsize=15)
date = ['11月8日', '11月14日', '11月22日', '11月29日', '12月6日']
dt = list(range(len(series[0:720])))
plt.xticks(range(1, len(dt), 7*24), date, rotation=0)
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/5-4-7.svg', format='svg')
plt.show()

# 尝试不同的取值进行对比

t_time = 720
t_time_val = 552
split_time = 360  # 可以尝试600 300
time_train = time[t_time_val - split_time:t_time_val]
x_train = series[t_time_val - split_time:t_time_val]
time_valid = time[t_time_val:t_time]
x_valid = series[t_time_val:t_time]
time_test = time[t_time:]
X_test = series[t_time:]

window_size = 100  # 可以尝试4 12 24 100
batch_size = 32  # 2^n

if split_time % 3 == 0:
    shuffle_buffer_size = np.float64(split_time / 3)
else:
    shuffle_buffer_size = np.float64(split_time / 2)


# shuffle_buffer_size = 200 #用来打乱数据集中数据顺序，是做随机采样使用的缓冲大小，此时设置的batchsize是buffer_size中输出的大小

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

# 模型训练

import time
start_time=time.process_time()
tf.keras.backend.clear_session()
#tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)#得到经过时间窗切片的滑动窗口训练数据
valid_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)#得到经过时间窗切片的滑动窗口训练数据
valid_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)#得到经过时间窗切片的滑动窗口训练数据
test_set = windowed_dataset(X_test, window_size, batch_size, shuffle_buffer=shuffle_buffer_size)#得到经过时间窗切片的滑动窗口训练数据

# 定义模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)#优化器：SGD 学习率1e-5 （可调节）
model.compile(loss=tf.keras.losses.Huber(),#使用上文定义的基于LSTM的model结构，loss函数选为Huber损失，即平滑的平均绝对误差
              optimizer=optimizer,
              metrics=["mae"])
class ValidationLossHistory(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.evaluate(self.validation_data)
        self.val_losses.append(val_loss)
        print(f'Epoch {epoch+1} - Validation loss: {val_loss}')

# 创建自定义回调函数实例
validation_loss_history = ValidationLossHistory(validation_data=valid_set)

history = model.fit(train_set,epochs=100,validation_data=valid_set,callbacks=[validation_loss_history])#训练伦次epochs (可调节)
model.summary()


# 观察损失值随训练次数的变化情况

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and chpter5_code data
# sets for each training epoch
#-----------------------------------------------------------
loss=history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(loss)) # Get number of epochs
epochs_val = range(len(val_loss))

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure()
plt.plot(epochs, loss, 'r', label='训练损失')
plt.plot(epochs_val, val_loss, 'g', label='验证损失')
plt.title('LSTM')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('./results/5-4-8.svg', format='svg')
plt.show()

# 观察模型预测运行的时间

end_time=time.process_time()
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[t_time - window_size:-1, -1, 0]
test_forecast = rnn_forecast[:-1]
test_time = time_test[:-1]
print('Running time: %s Seconds'%(int(end_time-start_time)))

# 观察预测的流量值以及RMSE

plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

#plt.plot(time_train, x_train, label='Train')

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
plt.savefig('./results/5-4-9.svg', format='svg')
plt.show()

RMSE =mean_squared_error(X_test, rnn_forecast)**0.5
MAE = mean_absolute_error(X_test, rnn_forecast)
R2 = r2_score(X_test, rnn_forecast)
print('LSTM -> MAE: %f.  RMSE: %f.  R2_score: %f.' % (RMSE, MAE, R2))
