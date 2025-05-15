import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子保证可重复性
np.random.seed(42)

# Lorenz系统参数设置
T = 1000  # 总序列长度（时间步数）
dt = 0.01  # 时间步长（数值积分步长）
sigma = 10.0  # 经典Lorenz参数
rho = 28.0    # 经典Lorenz参数
beta = 8/3    # 经典Lorenz参数

def generate_lorenz_system(T=1000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, x0=0.1, y0=0.1, z0=0.1):
    """
    生成Lorenz系统时间序列（使用欧拉法）
    参数：
        T : 总序列长度（时间步数）
        dt : 数值积分步长
        sigma, rho, beta : Lorenz系统参数
        x0, y0, z0 : 初始条件
    返回：
        x, y, z : 生成的x, y, z时间序列
    """
    n_steps = T
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    
    # 设置初始条件
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    # 欧拉法数值积分
    for t in range(n_steps - 1):
        x[t+1] = x[t] + dt * sigma * (y[t] - x[t])
        y[t+1] = y[t] + dt * (x[t] * (rho - z[t]) - y[t])
        z[t+1] = z[t] + dt * (x[t] * y[t] - beta * z[t])
    
    return x, y, z

# 生成时间索引列
time_index = np.arange(T)

# 参数组合设置（探索不同的rho值以观察不同行为）
rho_values = [28.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0, 0.0]  # 不同rho参数，sigma和beta固定
# rho=28.0 → 经典混沌状态
# rho<13.0 → 可能出现周期或收敛行为

# 生成数据集
datasets = [time_index]  # 第一列为时间索引

for rho in rho_values:
    # 生成不同rho值的Lorenz系统序列（仅保存x序列，y和z可类似处理）
    x_series, _, _ = generate_lorenz_system(T=T, dt=dt, sigma=10.0, rho=rho, beta=8/3)
    datasets.append(x_series)

# 创建DataFrame
columns = ['date'] + [f'x{tau + 1}' for tau in range(8)]
df = pd.DataFrame(np.array(datasets).T, columns=columns)

# 数据标准化（可选）
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns[1:]] = scaler.fit_transform(df[columns[1:]])

# 保存为CSV文件
csv_file_path = './data_sets/LorenzSystem.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
df.to_csv(csv_file_path, index=False)

print(f"Lorenz系统数据已保存到 {csv_file_path}")
print(f"数据维度: {df.shape} (时间步长×特征数)")
print("前5行数据示例:")
print(df.head())

# 可视化验证：时间序列
plt.figure(figsize=(12, 6))
for col in columns[1:]:
    plt.plot(df['date'][:200], df[col][:200], label=col)
plt.title("Lorenz系统时间序列示例(前200时间步)")
plt.xlabel("时间步")
plt.ylabel("标准化值")
plt.legend()
plt.grid(True)
plt.show()
