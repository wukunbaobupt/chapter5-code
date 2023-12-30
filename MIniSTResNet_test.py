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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings("ignore")
print(tensorflow.__version__)

# 一、数据预处理

# 1.1 数据导入（加载)
# 定义数据的位置
# 2013-11、2013-12是米兰市100*100网络中心的20*20的网络数据
# 2013-11-fusion、2013-12-fusion是将100*100网络聚合成20*20网络之后的数据
total_data_path = '../Data/total.vocab'
data_11 =  './data/2013-11-fusion.vocab'
data_12 = './data/2013-12-fusion.vocab'
max_min_path = '../Data/loc_max_mix.vocab'
with open(data_11,"r") as f:  #设置文件对象
    print(f.readline().strip())

# 1.2 处理缺失值(零值)
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
OPTIMIZER = optimizers.SGD(lr=0.03)
LOSS = 'mean_squared_error'
BATCH_SIZE = 64
EPOCHS = 50


# 设置输入数据的格式，closeness、period、trend分别对应人工特征工程下时间维度的特征抽取后，准备放入模型的数据格式
closeness = layers.Input(shape=(20, 20, 3), name='closeness')
period = layers.Input(shape=(20, 20, 3), name='period')
trend = layers.Input(shape=(20, 20, 1), name='trend')
convlstmall = layers.Input(shape=(20,20, ),name='convlstmall')  #convLSTM的完整输入数据的格式
#数据类型是空间栅格，实例和格式都是以Matrics存储栅格数据 -choice A

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
DataPreProcess.SaveLabelData(test_label, '../Data/labels.vocab')
STEPS_PER_EPOCH = int(train_data.shape[0] // BATCH_SIZE)

#自定义回调函数(用于计算验证损失)
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

# 2.2.2 MiniSTResNet模型的训练和预测

MiniSTResNet_output =BuildModel.MiniSTResNet(closeness, period, trend, filters=64, kernel_size=(3,3), activation='relu', use_bias=True)
MiniSTResNet = Model(inputs=[closeness, period, trend], outputs=MiniSTResNet_output)
MiniSTResNet.compile(optimizer=OPTIMIZER,loss=LOSS,metrics=None)
# 去掉下面一句statement的注释可以显示整个模型结构
MiniSTResNet.summary()

# 训练和保存模型
print('-> Start to train MiniSTResNet model!')
MiniSTResNet_history = MiniSTResNet.fit_generator(generator=TrainModel.Generator([c_train, p_train, t_train], train_label,  STEPS_PER_EPOCH, BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=([c_valid, p_valid, t_valid],valid_label), callbacks=[validation_loss_history], verbose=1, shuffle=True)
MiniSTResNet_path = '../Model/MiniSTResNet.h5'
MiniSTResNet.save_weights(MiniSTResNet_path)
print('-> Finish to save MiniSTResNet model')

# 保存预测数据
MiniSTResNet_result_path = '../Data/MiniSTResNet_result.vocab'
MiniSTResNet_predict = MiniSTResNet.predict([c_test, p_test, t_test], batch_size=BATCH_SIZE ,verbose=1)
TrainModel.WriteData(MiniSTResNet_result_path, MiniSTResNet_predict)

# 3.1 Loss可视化
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(MiniSTResNet_history.history['loss'], color='r', label='训练损失')
plt.plot(MiniSTResNet_history.history['val_loss'], color='g', label='验证损失')
plt.title('MiniSTResNet')
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('../result/5-5-9.svg', format='svg')
plt.show()

MiniSTResNet_result_path = '../Data/MiniSTResNet_result.vocab'
max_min_path = '../Data/loc_max_mix.vocab'
label_path = '../Data/labels.vocab'

MiniSTResNet_result = Visualization.DecodeData(MiniSTResNet_result_path, max_min_path)
label = Visualization.DecodeData(label_path, max_min_path)

# 显示预测某区域的预测曲线和真实曲线
loc_id = 200
plt.figure(figsize=(15, 7))

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(np.arange(len(MiniSTResNet_result[:,loc_id])),label[:,loc_id], c='b', label='真实值')
plt.plot(np.arange(len(MiniSTResNet_result[:,loc_id])), MiniSTResNet_result[:,loc_id], c='r', label='预测值')
plt.plot(np.arange(len(MiniSTResNet_result[:,loc_id])), MiniSTResNet_result[:,loc_id] - label[:,loc_id], c='k', label='误差值')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
date = ['12月8日', '12月9日', '12月10日', '12月11日', '12月12日', '12月13日', '12月14日']
dt = list(range(len(label[:,loc_id])))
plt.xticks(range(1, len(dt), 24), date, rotation=0)
plt.legend(loc='best', fontsize=15)
plt.title("MiniSTResNet 预测值、实际值和误差分布图 ID=%d "%(loc_id), fontsize=20)
plt.xlabel('时间', fontsize=15)
plt.ylabel('流量', fontsize=15)
plt.tight_layout()
plt.savefig('../result/MiniSTResNet.svg', format='svg')
plt.show()

#计算性能
MiniSTResNet_RMSE =mean_squared_error(MiniSTResNet_result[:,loc_id], label[:,loc_id])**0.5
MiniSTResNet_MAE = mean_absolute_error(MiniSTResNet_result[:,loc_id], label[:,loc_id])
MiniSTResNet_R2_score = r2_score(MiniSTResNet_result[:,loc_id], label[:,loc_id])

print('MiniSTResNet -> MAE: %f.  RMSE: %f.  R2_score: %f.' % (MiniSTResNet_MAE, MiniSTResNet_RMSE, MiniSTResNet_R2_score))

