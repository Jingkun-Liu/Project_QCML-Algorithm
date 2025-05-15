import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
# from model import LSTM
from Model_quan import HybridModel

# 设置字体为SimHei，用于显示中文
plt.rcParams['font.family'] = 'SimHei'

# 设置随机种子
torch.manual_seed(42)

#  读取CSV文件
current_path = os.getcwd()
print("当前工作路径是:", current_path)
file_path = os.path.join(current_path, 'data_sets', 'MackeyGlass.csv')
save_path = os.path.join(current_path, 'data_sets', 'lstm_model.pth')
data = pd.read_csv(file_path)

# 确保日期列是 datetime 类型
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 选择多特征：% WEIGHTED ILI,%UNWEIGHTED ILI,AGE 0-4,AGE 5-24,ILITOTAL,NUM. OF PROVIDERS,OT
features = data[['x1','x2','x3','x4','x5','x6','x7','x8']].values

# 修改数据预处理部分（原代码需要修改）
# 正确划分：训练集80%，测试集20%
train_size = int(len(features) * 0.8)
train_features = features[:train_size]
test_features = features[train_size:]

# 只在训练集上拟合归一化器
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_features)
scaled_test = scaler.transform(test_features)  # 测试集用训练集的scaler转换

# 准备训练、验证和测试数据（原代码替换）
train_data = scaled_train
test_data = scaled_test

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        # b = data[i + time_step].unsqueeze(1)
        b = data[i + time_step]
        b = b.reshape(1,8)
        y.append(b)
    return np.array(X), np.array(y)

# 创建数据集
time_step = 10  # 时间步长
X_train, y_train = create_dataset(train_data, time_step)

# 转换为PyTorch张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# 封装为 TensorDataset
train_dataset = TensorDataset(X_train, y_train)

# 定义 DataLoader
batch_size = 32  # 可以根据需要调整 batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 100

# 初始化模型、损失函数和优化器
model = HybridModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # output1,output2 = model(batch_X)
        # loss1 = criterion(output1, batch_y)
        # loss2 = criterion(output2, batch_y)
        # loss = loss1 * alpha + loss2 * (1 - alpha)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
    
    # 每隔一定轮数打印损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), save_path)
print("模型已保存为 'lstm_model.pth'")

time_step = 10  # 时间步长

# 创建测试集
X_test, y_test = create_dataset(test_data, time_step)

# 转换为PyTorch张量
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# 封装为DataLoader（可选）
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载保存的模型
# model = HybridModel()
# model.load_state_dict(torch.load(save_path))
model.eval()  # 设置为评估模式

# 测试过程
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # output1,output2 = model(batch_X)
        # loss1 = criterion(output1, batch_y)
        # loss2 = criterion(output2, batch_y)
        # loss = loss1 * alpha + loss2 * (1 - alpha)
        # outputs = output1 * alpha + output2 * (1 - alpha)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        test_loss += loss.item()
        
        # 保存预测结果和实际值
        predictions.append(outputs.numpy())
        actuals.append(batch_y.numpy())

# 计算平均测试损失
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# 合并批次结果
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# 反归一化需要构造完整的时序数据
def inverse_scale_predictions(scaler, preds, actuals, time_step):
    # 假设每个预测步需要与前time_step步拼接进行反归一化
    dummy_data = np.zeros((len(preds) + time_step, scaler.n_features_in_))
    dummy_data[time_step:] = preds.reshape(-1, scaler.n_features_in_)
    inv_preds = scaler.inverse_transform(dummy_data)[time_step:]
    
    dummy_data[time_step:] = actuals.reshape(-1, scaler.n_features_in_)
    inv_actuals = scaler.inverse_transform(dummy_data)[time_step:]
    
    return inv_preds, inv_actuals

# 获取反归一化后的数据
inv_preds, inv_actuals = inverse_scale_predictions(scaler, predictions, actuals, time_step)

# 提取目标变量（例如第一个特征'% WEIGHTED ILI'）
target_idx = 0
predicted_target = inv_preds[:, target_idx]
actual_target = inv_actuals[:, target_idx]

plt.figure(figsize=(12,6))
plt.plot(actual_target, label='Actual Values')
plt.plot(predicted_target, label='Predicted Values')
plt.title('Test Set Predictions vs Actuals')
plt.xlabel('Time Step')
plt.ylabel('% WEIGHTED ILI')
plt.legend()
plt.show()

difference = inv_actuals - inv_preds
# 将 difference 转换为 DataFrame
difference_df = pd.DataFrame(difference, columns=['x1','x2','x3','x4','x5','x6','x7','x8'])
# 保存为 CSV 文件
difference_df.to_csv(f'difference__mackeyglass1000_quantum_{num_epochs}.csv', index=False)
