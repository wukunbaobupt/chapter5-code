import Visualization
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 获取预测数据并归一化
MiniDeepST_result_path = '../Data/MiniDeepST_result.vocab'
MiniSTResNet_result_path = '../Data/MiniSTResNet_result.vocab'
max_min_path = '../Data/loc_max_mix.vocab'
label_path = '../Data/labels.vocab'

MiniDeepST_result = Visualization.DecodeData(MiniDeepST_result_path, max_min_path)
MiniSTResNet_result = Visualization.DecodeData(MiniSTResNet_result_path, max_min_path)
label = Visualization.DecodeData(label_path, max_min_path)

# 构建预测结果字典
results_dict = {'MiniDeepST':MiniDeepST_result, 'MiniSTResNet': MiniSTResNet_result}

# 显示预测某区域的预测曲线和真实曲线
loc_id = 200
Visualization.ShowPrediction(loc_id, results_dict, label)

#计算性能
MiniDeepST_RMSE =mean_squared_error(MiniDeepST_result[:,loc_id], label[:,loc_id])**0.5
MiniDeepST_MAE = mean_absolute_error(MiniDeepST_result[:,loc_id], label[:,loc_id])
MiniDeepST_R2_score = r2_score(MiniDeepST_result[:,loc_id], label[:,loc_id])

MiniSTResNet_RMSE =mean_squared_error(MiniSTResNet_result[:,loc_id], label[:,loc_id])**0.5
MiniSTResNet_MAE = mean_absolute_error(MiniSTResNet_result[:,loc_id], label[:,loc_id])
MiniSTResNet_R2_score = r2_score(MiniSTResNet_result[:,loc_id], label[:,loc_id])

print('MiniDeepST   -> MAE: %f.  RMSE: %f.  R2_score: %f.' % (MiniDeepST_MAE, MiniDeepST_RMSE, MiniDeepST_R2_score))
print('MiniSTResNet -> MAE: %f.  RMSE: %f.  R2_score: %f.' % (MiniSTResNet_MAE, MiniSTResNet_RMSE, MiniSTResNet_R2_score))

# 显示热力图
hour = 12
Visualization.HotMap(hour, results_dict, label)