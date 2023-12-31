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

# %load DataPreProcess.py
# 2.1 DeepST 模型的剪裁设计（空间CNN自动特征工程与时序人工特征工程的方法）

# 超参数
OPTIMIZER = optimizers.SGD(lr=0.03)
LOSS = 'mean_squared_error'
BATCH_SIZE = 16
EPOCHS = 50


# 数据集处理
# 获取预处理后的数据
train_data, valid_data, test_data = TrainModel.MakeDataset(total_data_path)

# 获取训练数据
c_train = train_data[:, :, :, 0:3]
p_train = train_data[:, :, :, 3:6]
t_train = train_data[:, :, :, 6:7]
train_label = train_data[:, :, :, -1]
train_label = np.reshape(train_label, [train_data.shape[0], 400])

# 获取验证数据
c_valid = valid_data[:, :, :, 0:3]
p_valid = valid_data[:, :, :, 3:6]
t_valid = valid_data[:, :, :, 6:7]
valid_label = valid_data[:, :, :, -1]
valid_label = np.reshape(valid_label, [valid_data.shape[0], 400])

# 获取测试的数据
c_test = test_data[:, :, :, 0:3]
p_test = test_data[:, :, :, 3:6]
t_test = test_data[:, :, :, 6:7]
test_label = test_data[:, :, :, -1]
test_label = np.reshape(test_label, [test_data.shape[0], 400])
DataPreProcess.SaveLabelData(test_label, './Data/labels.vocab')
STEPS_PER_EPOCH = int(train_data.shape[0] // BATCH_SIZE)

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
validation_loss_history = ValidationLossHistory(validation_data=[c_valid, p_valid, t_valid])

# 设置输入数据的格式，closeness、period、trend分别对应人工特征工程下时间维度的特征抽取后，准备放入模型的数据格式
closeness = layers.Input(shape=(20, 20, 3), name='closeness')
period = layers.Input(shape=(20, 20, 3), name='period')
trend = layers.Input(shape=(20, 20, 1), name='trend')
convlstmall = layers.Input(shape=(20,20, ),name='convlstmall')  #convLSTM的完整输入数据的格式
#数据类型是空间栅格，实例和格式都是以Matrics存储栅格数据 -choice A

# 2.1.1 MiniDeepST
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

MiniDeepST_history = MiniDeepST.fit_generator(generator=TrainModel.Generator([c_train, p_train, t_train], train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=([c_valid, p_valid, t_valid], valid_label), callbacks=[validation_loss_history], verbose=1, shuffle=True)
MiniDeepST_path = './Model/MiniDeepST.h5'
MiniDeepST.save_weights(MiniDeepST_path)
print('-> Finish to save trained MiniDeepST model')

# 保存预测数据
MiniDeepST_result_path = './Data/MiniDeepST_result.vocab'
MiniDeepST_predict = MiniDeepST.predict([c_test, p_test, t_test], batch_size=BATCH_SIZE ,verbose=1)
print(MiniDeepST_predict.shape)
TrainModel.WriteData(MiniDeepST_result_path, MiniDeepST_predict)

# 3.1 Loss可视化

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(MiniDeepST_history.history['loss'], color='r', label='训练损失')
plt.plot(MiniDeepST_history.history['val_loss'], color='g', label='验证损失')
plt.title('MiniDeepST')
plt.ylabel('loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./results/5-5-4.svg', format='svg')
plt.show()


# 3.2 结果可视化

# 3.2.1 获取预测结果

# 获取预测数据并归一化
MiniDeepST_result_path = './Data/MiniDeepST_result.vocab'
max_min_path = './Data/loc_max_mix.vocab'
label_path = './Data/labels.vocab'

MiniDeepST_result = Visualization.DecodeData(MiniDeepST_result_path, max_min_path)
label = Visualization.DecodeData(label_path, max_min_path)

# 显示预测某区域的预测曲线和真实曲线
loc_id = 200
plt.figure(figsize=(15, 7))

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(np.arange(len(MiniDeepST_result[:,loc_id])),label[:,loc_id], c='b', label='真实值')
plt.plot(np.arange(len(MiniDeepST_result[:,loc_id])), MiniDeepST_result[:,loc_id], c='r', label='预测值')
plt.plot(np.arange(len(MiniDeepST_result[:,loc_id])), MiniDeepST_result[:,loc_id] - label[:,loc_id], c='k', label='误差值')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
dt = list(range(len(label[:,loc_id])))
plt.xticks(range(1, len(dt), 24), date, rotation=0)
# legend设置图例
plt.legend(loc='best', fontsize=15)
plt.title("MiniDeepST 预测值、实际值和误差分布图 ID=%d "%(loc_id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.tight_layout()
plt.savefig('./results/5-5-5.svg', format='svg')
plt.show()

#计算性能

MiniDeepST_RMSE =mean_squared_error(MiniDeepST_result[:,loc_id], label[:,loc_id])**0.5
MiniDeepST_MAE = mean_absolute_error(MiniDeepST_result[:,loc_id], label[:,loc_id])
MiniDeepST_R2_score = r2_score(MiniDeepST_result[:,loc_id], label[:,loc_id])

print('MiniDeepST   -> MAE: %f.  RMSE: %f.  R2_score: %f.' % (MiniDeepST_MAE, MiniDeepST_RMSE, MiniDeepST_R2_score))

