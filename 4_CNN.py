from keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import DataPreProcess
import BuildModel
import TrainModel
import Visualization
import tensorflow
import warnings
from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import Callback
warnings.filterwarnings("ignore")
print(tensorflow.__version__)

# 数据导入
# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据
total_data_path = './Data/total.vocab'
data_11 = './Data/2013-11-fusion.vocab'
data_12 = './Data/2013-12-fusion.vocab'
max_min_path = './Data/loc_max_mix.vocab'

# 处理缺失值
data_without_missing_value = DataPreProcess.ProcessMissingValue(data_11, data_12, city_amount=400, judge_num=7)

# 处理异常值
data_without_abnormal_value = DataPreProcess.ProcessAbnormalValue(data_without_missing_value, city_amount=400, judge_week_num=8, judge_day_num=30)

# 归一化数据
total_data = DataPreProcess.DataNormalization(data_without_abnormal_value, max_min_path, city_amount=400)

# 数据保存
DataPreProcess.SavePreProcessData(total_data, total_data_path)

# 超参数
OPTIMIZER = optimizers.Adam(lr=5e-5)
LOSS = tensorflow.keras.losses.Huber()
BATCH_SIZE = 32
EPOCHS = 100

tensorflow.random.set_seed(47)
np.random.seed(47)

# 设置输入数据的格式，closeness、period、trend分别对应人工特征工程下时间维度的特征抽取后，准备放入模型的数据格式
closeness = layers.Input(shape=(20, 20, 3), name='closeness')
period = layers.Input(shape=(20, 20, 3), name='period')
trend = layers.Input(shape=(20, 20, 1), name='trend')
convlstmall = layers.Input(shape=(20,20, ),name='convlstmall')  #convLSTM的完整输入数据的格式
#数据类型是空间栅格，实例和格式都是以Matrics存储栅格数据 -choice A

# 数据集处理
# 获取预处理后的数据
train_data, test_data = TrainModel.MakeDataset(total_data_path)

# 获取训练数据
c_train = train_data[:, :, :, 0:3]
p_train = train_data[:, :, :, 3:6]
t_train = train_data[:, :, :, 6:7]
train_label = train_data[:, :, :, -1]
train_label = np.reshape(train_label, [train_data.shape[0], 400])

# 获取测试的数据
c_test = test_data[:, :, :, 0:3]
p_test = test_data[:, :, :, 3:6]
t_test = test_data[:, :, :, 6:7]
test_label = test_data[:, :, :, -1]
test_label = np.reshape(test_label, [test_data.shape[0], 400])
DataPreProcess.SaveLabelData(test_label, './Data/labels.vocab')
STEPS_PER_EPOCH = int(train_data.shape[0] // BATCH_SIZE)

# def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
#     # 函数功能：制作基于滑动窗口的数据集，将数据集按照窗口大小和步长进行切分，并
#     #          将所有窗口数据展平成一个大的数据集，对数据集随机采样，最后将每个
#     #          样本划分为训练数据和目标数据。
#     #    输入：series—输入训练数据
#     #          window_size—滑动窗口的大小，即每个样本包含多少个连续时间步的数据
#     #          batch_size—数据批量大小，即每次训练模型时输入的样本数量
#     #          shuffle_buffer—做随机采样使用的缓冲大小，用来打乱数据集中数据顺序
#     #    输出：ds.batch(batch_size).prefetch(1) —大小为batch_size的数据集
#
#     series = tensorflow.expand_dims(series, axis=-1)  # 输入训练数据进行一维展平
#     ds = tensorflow.data.Dataset.from_tensor_slices(series)
#     ds = ds.window(window_size + 1, shift=1, drop_remainder=True)  # 将数据集按照窗口大小和步长进行切分
#     ds = ds.flat_map(lambda w: w.batch(window_size + 1))
#     ds = ds.shuffle(shuffle_buffer)  # 对数据集进行随机采样，以防止训练过程中的过拟合
#     ds = ds.map(lambda w: (w[:-1], w[1:]))  # 将每个样本划分为训练数据和目标数据。
#     return ds.batch(batch_size).prefetch(1)


# 1 MiniDeepST
# mini-DeepST模型构建
MiniDeepST_output = BuildModel.MiniDeepST(closeness, period, trend, filters=64, kernel_size=(3,3), activation='relu', use_bias=True)
MiniDeepST = Model(inputs=[closeness, period, trend], outputs=MiniDeepST_output)
MiniDeepST.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
# 去掉下面一句statement的注释可以显示整个模型结构
MiniDeepST.summary()

# 2.2 DeepST修剪模型的训练和预测
# 2.2.1 MiniDeepST模型的训练和预测
# 训练和保存模型
print('-> Start to train MiniDeepST model!')

MiniDeepST_history = MiniDeepST.fit_generator(generator=TrainModel.Generator([c_train, p_train, t_train], train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
MiniDeepST_path = './Model/MiniDeepST.h5'
MiniDeepST.save_weights(MiniDeepST_path)
print('-> Finish to save trained MiniDeepST model')

# 保存预测数据
MiniDeepST_result_path = './Data/MiniDeepST_result.vocab'
MiniDeepST_predict = MiniDeepST.predict([c_test, p_test, t_test], batch_size=BATCH_SIZE ,verbose=1)
print(MiniDeepST_predict.shape)
TrainModel.WriteData(MiniDeepST_result_path, MiniDeepST_predict)


# 2.2.2 MiniSTResNet模型的训练和预测
MiniSTResNet_output =BuildModel.MiniSTResNet(closeness, period, trend, filters=64, kernel_size=(3,3), activation='relu', use_bias=True)
MiniSTResNet = Model(inputs=[closeness, period, trend], outputs=MiniSTResNet_output)
MiniSTResNet.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
MiniSTResNet.summary()
# 去掉下面一句statement的注释可以显示整个模型结构
# 训练和保存模型
print('-> Start to train MiniSTResNet model!')
MiniSTResNet_history = MiniSTResNet.fit_generator(generator=TrainModel.Generator([c_train, p_train, t_train], train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
MiniSTResNet_path = './Model/MiniSTResNet.h5'
MiniSTResNet.save_weights(MiniSTResNet_path)
print('-> Finish to save MiniSTResNet model')

# 保存预测数据
MiniSTResNet_result_path = './Data/MiniSTResNet_result.vocab'
MiniSTResNet_predict = MiniSTResNet.predict([c_test, p_test, t_test], batch_size=BATCH_SIZE ,verbose=1)
TrainModel.WriteData(MiniSTResNet_result_path, MiniSTResNet_predict)


# 2.2.2 MiniSTResNet_dropout_dropout模型的训练和预测
MiniSTResNet_dropout_output =BuildModel.MiniSTResNet_dropout(closeness, period, trend, filters=64, kernel_size=(3,3), activation='relu', use_bias=True, dropout=0.25)
MiniSTResNet_dropout = Model(inputs=[closeness, period, trend], outputs=MiniSTResNet_dropout_output)
MiniSTResNet_dropout.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
MiniSTResNet_dropout.summary()
# 去掉下面一句statement的注释可以显示整个模型结构
# 训练和保存模型
print('-> Start to train MiniSTResNet_dropout model!')
MiniSTResNet_dropout_history = MiniSTResNet_dropout.fit_generator(generator=TrainModel.Generator([c_train, p_train, t_train], train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
MiniSTResNet_dropout_path = './Model/MiniSTResNet_dropout.h5'
MiniSTResNet_dropout.save_weights(MiniSTResNet_dropout_path)
print('-> Finish to save MiniSTResNet_dropout model')

# 保存预测数据
MiniSTResNet_dropout_result_path = './Data/MiniSTResNet_dropout_result.vocab'
MiniSTResNet_dropout_predict = MiniSTResNet_dropout.predict([c_test, p_test, t_test], batch_size=BATCH_SIZE ,verbose=1)
TrainModel.WriteData(MiniSTResNet_dropout_result_path, MiniSTResNet_dropout_predict)


# 模型训练完毕后，清理 TensorFlow 默认图和相关资源
tensorflow.keras.backend.clear_session()


# 3.1 Loss可视化
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(MiniDeepST_history.history['loss'], color='r', label='MiniDeepST')
plt.plot(MiniSTResNet_history.history['loss'], color='orange', label='MiniSTResNet')
plt.plot(MiniSTResNet_dropout_history.history['loss'], color='g', label='MiniSTResNet_Dropout')
plt.title('训练损失')
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./results/5-5-9.svg', format='svg')
plt.show()

MiniDeepST_result_path = './Data/MiniDeepST_result.vocab'
MiniSTResNet_result_path = './Data/MiniSTResNet_result.vocab'
MiniSTResNet_dropout_result_path = './Data/MiniSTResNet_dropout_result.vocab'

max_min_path = './Data/loc_max_mix.vocab'
label_path = './Data/labels.vocab'

MiniDeepST_result = Visualization.DecodeData(MiniDeepST_result_path, max_min_path)
MiniSTResNet_result = Visualization.DecodeData(MiniSTResNet_result_path, max_min_path)
MiniSTResNet_dropout_result = Visualization.DecodeData(MiniSTResNet_dropout_result_path, max_min_path)
label = Visualization.DecodeData(label_path, max_min_path)

# 构建预测结果字典
results_dict = {'MiniDeepST': MiniDeepST_result, 'MiniSTResNet': MiniSTResNet_result, 'MiniSTResNet_Dropout': MiniSTResNet_dropout_result}

# 显示预测某区域的预测曲线和真实曲线
loc_id = 200
plt.figure(figsize=(15, 7))

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(np.arange(len(MiniSTResNet_dropout_result[:,loc_id])),label[:,loc_id], c='b', label='真实值')
plt.plot(np.arange(len(MiniDeepST_result[:,loc_id])), MiniDeepST_result[:,loc_id], c='r', label='MiniDeepST')
plt.plot(np.arange(len(MiniSTResNet_result[:,loc_id])), MiniSTResNet_result[:,loc_id], c='orange', label='MiniSTResNet')
plt.plot(np.arange(len(MiniSTResNet_dropout_result[:,loc_id])), MiniSTResNet_dropout_result[:,loc_id], c='g', label='MiniSTResNet_Dropout')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
dt = list(range(len(label[:,loc_id])))
plt.xticks(range(1, len(dt), 24), date, rotation=0)
plt.legend(loc='best', fontsize=15)
plt.title("预测值、实际值分布图 ID=%d "%(loc_id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-5-10.svg', format='svg')
plt.show()

# 显示热力图,预测结果的空间可视化
hour = 12
Visualization.HotMap(hour, results_dict, label)

#计算性能
MiniDeepST_RMSE =Visualization.CalculateRMSE(MiniDeepST_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniDeepST_MAE =Visualization.CalculateMAE(MiniDeepST_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniDeepST_R2_score =Visualization.CalculateR2score(MiniDeepST_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))

MiniSTResNet_RMSE =Visualization.CalculateRMSE(MiniSTResNet_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniSTResNet_MAE =Visualization.CalculateMAE(MiniSTResNet_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniSTResNet_R2_score =Visualization.CalculateR2score(MiniSTResNet_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))

MiniSTResNet_dropout_RMSE =Visualization.CalculateRMSE(MiniSTResNet_dropout_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniSTResNet_dropout_MAE =Visualization.CalculateMAE(MiniSTResNet_dropout_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))
MiniSTResNet_dropout_R2_score =Visualization.CalculateR2score(MiniSTResNet_dropout_result[:,loc_id].reshape((-1, 1)), label[:,loc_id].reshape((-1, 1)))

print('MiniDeepST   -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (MiniDeepST_RMSE, MiniDeepST_MAE, MiniDeepST_R2_score))
print('MiniSTResNet -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (MiniSTResNet_RMSE, MiniSTResNet_MAE, MiniSTResNet_R2_score))
print('MiniSTResNet_Dropout -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (MiniSTResNet_dropout_RMSE, MiniSTResNet_dropout_MAE, MiniSTResNet_dropout_R2_score))
