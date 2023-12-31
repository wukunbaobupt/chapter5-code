import keras

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
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import Callback


warnings.filterwarnings("ignore")
print(tensorflow.__version__)

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

# 1.4 归一化

# 归一化数据
total_data = DataPreProcess.DataNormalization(data_without_abnormal_value, max_min_path, city_amount=400)

# 1.5 保存可喂入学习模型的数据

# 数据保存
DataPreProcess.SavePreProcessData(total_data, total_data_path)

# 超参数
OPTIMIZER = optimizers.SGD(lr=0.01)
LOSS = 'mean_squared_error'
BATCH_SIZE = 8
EPOCHS = 10

# 设置输入数据的形式
data = layers.Input(shape=(20, 20, 168), name='data')
# 获取预处理后的数据
# total_data_path = './data/total.vocab'
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

ConvLSTMModel_output = BuildModel.ConvLSTM2(data)
ConvLSTMModel = Model(inputs=[data], outputs=ConvLSTMModel_output)
ConvLSTMModel.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
# 去掉注释可以显示整个模型结构
ConvLSTMModel.summary()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二，三块GPU（从0开始）
# 训练和保存模型
print('-> Start to train ConvLSTM model!')
ConvLSTMModel_history = ConvLSTMModel.fit_generator(generator= TrainModel.GeneratorConvLSTM(input_train, train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, shuffle=True)
ConvLSTMModel_path = './model/ConvLSTMModel.h5'
ConvLSTMModel.save_weights(ConvLSTMModel_path)
print('-> Finish to save ConvLSTM model')

# 保存预测数据
ConvLSTMModel_result_path = './Data/ConvLSTMModel_result.vocab'
ConvLSTMModel_predict = ConvLSTMModel.predict(input_test, batch_size=BATCH_SIZE ,verbose=1)
print(ConvLSTMModel_predict.shape)
TrainModel.WriteData(ConvLSTMModel_result_path, ConvLSTMModel_predict)

# 3.1 Loss可视化

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(ConvLSTMModel_history.history['loss'], color='r', label='训练损失')
plt.title('Mioss', fontsize=15)
plt.xlabel('EpniDeepST')
plt.ylabel('loch', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./results/5-5-4.svg', format='svg')
plt.show()



