import keras

from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import DataPreProcess
import BuildModel
import TrainModel
import Visualization
import tensorflow
import warnings
from tensorflow.keras import optimizers
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import Callback


warnings.filterwarnings("ignore")
print(tensorflow.__version__)

# 数据导入
# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据

total_data_path = './Data/total.vocab'
data_11 =  './Data/2013-11-fusion.vocab'
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
OPTIMIZER = optimizers.Adam(lr=0.0003)
LOSS = tensorflow.keras.losses.Huber()
BATCH_SIZE = 32
EPOCHS = 175
# 设置输入数据的形式
data = layers.Input(shape=(20, 20, 168), name='data')
# 获取预处理后的数据
train_data, test_data = TrainModel.MakeConvLSTMDataset(total_data_path)

# 获取训练数据
input_train = train_data[:, :, :, :-1]
train_label = train_data[:, :, :, -1]
train_label = np.reshape(train_label, [train_data.shape[0], 400])

# 获取测试的数据
input_test = test_data[:, :, :, :-1]
test_label = test_data[:, :, :, -1]
test_label = np.reshape(test_label, [test_data.shape[0], 400])
STEPS_PER_EPOCH = int(train_data.shape[0] // BATCH_SIZE)

# 获取可见的设备列表
visible_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if visible_devices:
    # 设置 TensorFlow 只使用第一个 GPU
    tensorflow.config.experimental.set_visible_devices(visible_devices[0], 'GPU')
    print("TensorFlow is using GPU")
else:
    print("No GPU available")

#ConvLSTM 模型#

ConvLSTMModel_output = BuildModel.ConvLSTM2(data)
ConvLSTMModel = Model(inputs=[data], outputs=ConvLSTMModel_output)
ConvLSTMModel.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
# 去掉注释可以显示整个模型结构
ConvLSTMModel.summary()
# 训练和保存模型
print('-> Start to train ConvLSTM model!')
ConvLSTMModel_history = ConvLSTMModel.fit_generator(generator= TrainModel.GeneratorConvLSTM(input_train, train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
ConvLSTMModel_path = './Model/ConvLSTMModel.h5'
ConvLSTMModel.save_weights(ConvLSTMModel_path)
print('-> Finish to save ConvLSTM model')

# 保存预测数据
ConvLSTMModel_result_path = './Data/ConvLSTMModel_result.vocab'
ConvLSTMModel_predict = ConvLSTMModel.predict(input_test, batch_size=BATCH_SIZE ,verbose=1)
print(ConvLSTMModel_predict.shape)
TrainModel.WriteData(ConvLSTMModel_result_path, ConvLSTMModel_predict)

#ConvLSTM2_Dropout 模型#


ConvLSTM2_Dropout_output = BuildModel.ConvLSTM2_Dropout(data, dropout=0.05)
ConvLSTM2_Dropout = Model(inputs=[data], outputs=ConvLSTM2_Dropout_output)
ConvLSTM2_Dropout.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
# 去掉注释可以显示整个模型结构
ConvLSTM2_Dropout.summary()
# 训练和保存模型
print('-> Start to train ConvLSTM2_Dropout model!')
ConvLSTM2_Dropout_history = ConvLSTM2_Dropout.fit_generator(generator= TrainModel.GeneratorConvLSTM(input_train, train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
ConvLSTM2_Dropout_path = './Model/ConvLSTM2_Dropout.h5'
ConvLSTM2_Dropout.save_weights(ConvLSTM2_Dropout_path)
print('-> Finish to save ConvLSTM2_Dropout model')

# 保存预测数据
ConvLSTM2_Dropout_result_path = './Data/ConvLSTM2_Dropout_result.vocab'
ConvLSTM2_Dropout_predict = ConvLSTM2_Dropout.predict(input_test, batch_size=BATCH_SIZE ,verbose=1)
print(ConvLSTM2_Dropout_predict.shape)
TrainModel.WriteData(ConvLSTM2_Dropout_result_path, ConvLSTM2_Dropout_predict)


# 3.1 Loss可视化

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(ConvLSTMModel_history.history['loss'], color='r', label='ConvLST 训练损失')
plt.plot(ConvLSTM2_Dropout_history.history['loss'], color='g', label='ConvLSTM2_Dropout 训练损失')
plt.title('ConvLSTM ', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./results/ConvLSTM_loss.svg', format='svg')
plt.show()

ConvLSTM2_Dropout_result_path = './Data/ConvLSTM2_Dropout_result.vocab'
ConvLSTMModel_result_path = './Data/ConvLSTMModel_result.vocab'
max_min_path = './Data/loc_max_mix.vocab'
label_path = './Data/labels.vocab'

# 模型训练完毕后，清理 TensorFlow 默认图和相关资源
tensorflow.keras.backend.clear_session()

ConvLSTMModel_result = Visualization.DecodeData(ConvLSTMModel_result_path, max_min_path)
ConvLSTM2_Dropout_result = Visualization.DecodeData(ConvLSTM2_Dropout_result_path, max_min_path)
label = Visualization.DecodeData(label_path, max_min_path)

# 显示预测某区域的预测曲线和真实曲线
loc_id = 200
plt.figure(figsize=(15, 7))

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(np.arange(len(ConvLSTMModel_result[:,loc_id])),label[:,loc_id], c='b', label='真实值')
plt.plot(np.arange(len(ConvLSTMModel_result[:,loc_id])), ConvLSTMModel_result[:,loc_id], c='r', label='ConvLSTM 预测值')
plt.plot(np.arange(len(ConvLSTM2_Dropout_result[:,loc_id])), ConvLSTM2_Dropout_result[:,loc_id], c='y', label='ConvLSTM_Dropout 预测值')
# plt.plot(np.arange(len(ConvLSTMModel_result[:,loc_id])), ConvLSTMModel_result[:,loc_id] - label[:,loc_id], c='k', label='误差值')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
dt = list(range(len(label[:,loc_id])))
plt.xticks(range(1, len(dt), 24), date, rotation=0)
plt.legend(loc='best', fontsize=15)
plt.title("ConvLSTM 预测值、实际值分布图 ID=%d "%(loc_id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.tight_layout()
plt.savefig('./results/ConvLSTM.svg', format='svg')
plt.show()

# 计算性能
ConvLSTMModel_RMSE = mean_squared_error(ConvLSTMModel_result[:, loc_id], label[:, loc_id]) ** 0.5
ConvLSTMModel_MAE = mean_absolute_error(ConvLSTMModel_result[:, loc_id], label[:, loc_id])
ConvLSTMModel_R2_score = r2_score(ConvLSTMModel_result[:, loc_id], label[:, loc_id])

ConvLSTM2_Dropout_RMSE = mean_squared_error(ConvLSTM2_Dropout_result[:, loc_id], label[:, loc_id]) ** 0.5
ConvLSTM2_Dropout_MAE = mean_absolute_error(ConvLSTM2_Dropout_result[:, loc_id], label[:, loc_id])
ConvLSTM2_Dropout_R2_score = r2_score(ConvLSTM2_Dropout_result[:, loc_id], label[:, loc_id])

print('ConvLSTM -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (ConvLSTMModel_RMSE, ConvLSTMModel_MAE, ConvLSTMModel_R2_score))
print('ConvLSTM_Dropout -> RMSE: %f.  MAE: %f.  R2_score: %f.' % (ConvLSTM2_Dropout_RMSE, ConvLSTM2_Dropout_MAE, ConvLSTM2_Dropout_R2_score))



