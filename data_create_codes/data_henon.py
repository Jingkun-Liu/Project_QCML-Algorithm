import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子保证可重复性
np.random.seed(42)

# Hénon映射参数设置
T = 1000  # 总序列长度
a = 1.4   # 经典Hénon参数
b = 0.3   # 经典Hénon参数

def generate_henon_map(T=1000, a=1.4, b=0.3, x0=0, y0=0):
    """
    生成Hénon映射时间序列
    参数：
        T : 总序列长度
        a, b : Hénon映射参数
        x0, y0 : 初始条件
    返回：
        x_series, y_series : 生成的x和y时间序列
    """
    x = np.zeros(T)
    y = np.zeros(T)
    
    # 设置初始条件
    x[0] = x0
    y[0] = y0
    
    # 生成Hénon映射序列
    for t in range(T-1):
        x[t+1] = 1 - a * x[t]**2 + y[t]
        y[t+1] = b * x[t]
    
    return x, y

# 生成时间索引列
time_index = np.arange(T)

# 参数组合设置（探索不同的a值以观察不同行为）
a_values = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]  # 不同a参数，b固定为0.3
# a=1.4 → 经典混沌状态
# a<1.0 → 可能出现周期或收敛行为

# 生成数据集
datasets = [time_index]  # 第一列为时间索引

for a in a_values:
    # 生成不同a值的Hénon映射序列（仅保存x序列，y可类似处理）
    x_series, _ = generate_henon_map(T=T, a=a, b=0.3)
    datasets.append(x_series)

# 创建DataFrame
columns = ['date'] + [f'x{tau + 1}' for tau in range(8)]
df = pd.DataFrame(np.array(datasets).T, columns=columns)

# 数据标准化（可选）
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns[1:]] = scaler.fit_transform(df[columns[1:]])

# 保存为CSV文件
csv_file_path = './data_sets/HenonMap.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
df.to_csv(csv_file_path, index=False)

print(f"Hénon映射数据已保存到 {csv_file_path}")
print(f"数据维度: {df.shape} (时间步长×特征数)")
print("前5行数据示例:")
print(df.head())

# 可视化验证
plt.figure(figsize=(12, 6))
for col in columns[1:]:
    plt.plot(df['date'][:200], df[col][:200], label=col)
plt.title("Hénon映射时间序列示例(前200时间步)")
plt.xlabel("时间步")
plt.ylabel("标准化值")
plt.legend()
plt.grid(True)
plt.show()
