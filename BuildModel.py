# 导入库函数
import keras
import numpy
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

# 工具列表
########################################################
# 1. MiniDeepST
# 2. ResUnit
# 3. MiniSTResNet
# 4. MiniSTResNet+Dropout
# 5. ConvLSTMCell
# 6. ConvLSTM
# 7. History_Average
# 8. ConvLSTM+Dropout
########################################################

def MiniDeepST(closeness, period, trend, filters, kernel_size, activation, use_bias):    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(trend)
    fusion    = layers.Add()([closeness, period, trend])
    fusion    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion    = layers.Conv2D(1, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    res       = layers.Flatten(name='output')(fusion)
    return res

def ResUnit(data, filters, kernel_size):
    res = layers.ReLU()(data)
    res = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', use_bias=True)(res)
    res = layers.Conv2D(filters, kernel_size, padding='same', activation=None, use_bias=True)(res)
    res = layers.Add()([res,data])
    return res

def MiniSTResNet(closeness, period, trend, filters, kernel_size, activation, use_bias):    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = ResUnit(closeness, filters, kernel_size)
    period    = ResUnit(period, filters, kernel_size)
    trend     = ResUnit(trend, filters, kernel_size)

    closeness = ResUnit(closeness, filters, kernel_size)
    period    = ResUnit(period, filters, kernel_size)
    trend     = ResUnit(trend, filters, kernel_size)

    closeness = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(trend)
    
    fusion = layers.Add()([closeness,period,trend])
    fusion = layers.Conv2D(1, (1,1), padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion = layers.Flatten(name='output')(fusion)
    return fusion


def MiniSTResNet_dropout(closeness, period, trend, filters, kernel_size, activation, use_bias, dropout=0.5):
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)

    closeness = ResUnit(closeness, filters, kernel_size)
    period = ResUnit(period, filters, kernel_size)
    trend = ResUnit(trend, filters, kernel_size)

    closeness = ResUnit(closeness, filters, kernel_size)
    period = ResUnit(period, filters, kernel_size)
    trend = ResUnit(trend, filters, kernel_size)

    closeness = layers.Conv2D(filters, (1, 1), padding='same', activation=activation, use_bias=use_bias)(closeness)
    period = layers.Conv2D(filters, (1, 1), padding='same', activation=activation, use_bias=use_bias)(period)
    trend = layers.Conv2D(filters, (1, 1), padding='same', activation=activation, use_bias=use_bias)(trend)

    # 添加 Dropout  避免过拟合
    closeness = Dropout(dropout)(closeness)
    period = Dropout(dropout)(period)
    trend = Dropout(dropout)(trend)

    fusion = layers.Add()([closeness, period, trend])
    fusion = layers.Conv2D(1, (1, 1), padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion = layers.Flatten(name='output')(fusion)
    return fusion

def ConvLSTMCell(data):
    conv = layers.Conv2D(64, (3,3), padding='same', activation='relu', use_bias=True)(data)
    conv = ResUnit(conv, 64, (3,3))
    conv = ResUnit(conv, 64, (3,3))
    
    lstm_data = layers.Reshape((data.shape[-1], 20, 20, 1))(data)
    lstm = layers.ConvLSTM2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(lstm_data)
    
    lstm = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(lstm)
    conv = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(conv)
    res  = layers.Add()([lstm, conv])
    return res

def ConvLSTM(closeness, period, trend):
    closeness = ConvLSTMCell(closeness)
    period    = ConvLSTMCell(period)
    trend     = ConvLSTMCell(trend)
    closeness = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(closeness)
    period    = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(period)
    trend     = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(trend)
    res = layers.Add()([closeness,period,trend])
    res = layers.Conv2D(1, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Flatten(name='output')(res)
    return res
    
def ConvLSTM2(data):
    res = ConvLSTMCell(data)
    res = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Conv2D(1, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Flatten(name='output')(res)
    return res

def evaluate_naive_method(total_data):
    col = 20 #栅格数据的列数
    row = 20 #栅格数据的行数
    def HA(date,hour):
        counter = 0
        historical_average = 0
        while date>=1 and counter < 30:
            date -= 1
            historical_average += total_data[date,:,hour]
            counter += 1
        historical_average /= counter
        return historical_average
    batch_maes = []
    naive_method_predict = [[0] * col*row ] * 7 * 24
    naive_method_label = [[0] * col*row ] * 7 *24
    day_pos = 37 #预测的第一天在数据集中的相对位置
    for day in range(7):  #验证集部分 12月8日-12月24日共7天
        day_pos = 37 #预测的第一天在数据集中的相对位置
        #11月1日是一年中的第312天，12月8日是一年中的第352天,间隔37天
        for hour in range(24):
            preds = HA(day+day_pos,hour)
            labels = total_data[day+day_pos,:,hour]
            naive_method_predict[day * 24 + hour] = preds
            naive_method_label[day * 24 + hour] = labels           
            mae = numpy.mean(numpy.abs(preds-labels))
            batch_maes.append(mae)
    return numpy.array(naive_method_predict),numpy.array(naive_method_label)

def ConvLSTM2_Dropout(data,dropout = 0.1):
    res = ConvLSTMCell(data)
    res = Dropout(dropout)(res)
    res = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Conv2D(1, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Flatten(name='output')(res)
    return res