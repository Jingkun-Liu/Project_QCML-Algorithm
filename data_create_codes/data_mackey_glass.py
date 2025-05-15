import pandas as pd
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# 设置随机种子保证可重复性
np.random.seed(42)
torch.manual_seed(42)

# Mackey-Glass参数设置
T = 1000  # 总序列长度（建议至少包含10个周期）
tau = 17  # 延迟参数（经典混沌参数）
beta = 0.2  # 生产速率
gamma = 0.1  # 衰减速率
n = 10  # 非线性指数

def generate_mackey_glass(T=1000, tau=17, beta=0.2, gamma=0.1, n=10):
    """
    生成Mackey-Glass混沌时间序列
    参数：
        T : 总序列长度
        tau : 时间延迟（关键混沌参数）
        beta, gamma, n : 微分方程参数
    返回：
        mg_series : 生成的混沌时间序列
    """
    # 初始化序列（需要比tau长的初始化段）
    history_length = max(tau + 1, 100)
    x = np.zeros(T + history_length)
    
    # 初始条件设置（使用随机初始化）
    x[:history_length] = 1.5 + 0.2 * np.random.randn(history_length)
    
    # 使用欧拉法进行数值积分（步长Δt=1）
    for t in range(history_length, T + history_length - 1):
        x[t+1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**n) - gamma * x[t]
    
    # 去除初始过渡段，保留稳定状态
    return x[history_length:]

# 生成时间索引列（根据需要调整）
time_index = np.arange(T)

# 参数组合设置（主要改变tau值观察不同状态）
tau_values = [17, 23, 29, 35, 41, 47, 53, 60]  # 不同延迟参数
# tau=17 → 混沌状态
# tau=23 → 过渡状态
# tau=30 → 周期状态

# 生成数据集
datasets = [time_index]  # 第一列为时间索引

for tau in tau_values:
    # 生成不同tau值的时间序列
    mg_series = generate_mackey_glass(T=T, tau=tau)
    datasets.append(mg_series)

# 创建DataFrame
columns = ['date'] + [f'x{tau + 1}' for tau in range(8)]
df = pd.DataFrame(np.array(datasets).T, columns=columns)

# 数据标准化（可选）
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns[1:]] = scaler.fit_transform(df[columns[1:]])

# 保存为CSV文件
csv_file_path = './data_sets/MackeyGlass.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
df.to_csv(csv_file_path, index=False)

print(f"Mackey-Glass数据已保存到 {csv_file_path}")
print(f"数据维度: {df.shape} (时间步长×特征数)")
print("前5行数据示例:")
print(df.head())

# 可视化验证
plt.figure(figsize=(12, 6))
for col in columns[1:]:
    plt.plot(df['date'][:200], df[col][:200], label=col)
plt.title("Mackey-Glass时间序列示例(前200时间步)")
plt.xlabel("时间步")
plt.ylabel("标准化值")
plt.legend()
plt.grid(True)
plt.show()

