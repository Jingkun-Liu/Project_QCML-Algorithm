import pandas as pd
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
# from model import LSTM

# 设置字体为SimHei，用于显示中文
plt.rcParams['font.family'] = 'SimHei'

# 设置随机种子
torch.manual_seed(42)

T = 500  # 序列长度
tau = 10  # 正弦波的周期，可以根据需要调整
m = 0.1  # 趋势斜率，可以根据需要调整
sigma = 0.5  # 噪声标准差，可以根据需要调整
psi = np.random.uniform(0, 2 * np.pi)  # 从U(0,2π)中随机采样得到的相位

t = np.arange(T)  # 时间索引从0到T - 1
# 生成含噪趋势正弦波
x = np.sin((2 * np.pi / tau) * t + psi) + (m * t / T) + sigma * np.random.normal(size=T)

def generate_nts_dataset(tau, m, sigma):
    T = 500
    psi = np.random.uniform(0, 2 * np.pi)  # 随机相位
    t = np.arange(T)
    
    # 生成锯齿波（使用模运算生成线性斜坡）
    # 添加相位偏移并归一化到[0,1]范围
    sawtooth = (t + psi/(2*np.pi)*tau) % tau  # 相位偏移处理
    sawtooth = sawtooth / tau  # 标准化到[0,1)范围
    
    # 调整到[-1,1]范围（可选）
    # sawtooth = 2 * sawtooth - 1
    
    # 添加线性趋势和噪声
    signal = sawtooth + (m * t / T) + sigma * np.random.normal(size=T)
    return signal

# 单变量预测与多变量预测

# 生成6个不同参数设置的合成数据集
datasets = []
datasets.append(t)
tau_values = [8, 16]  # 不同的周期值
m_values = [0.1, 0.2]  # 不同的趋势斜率值
sigma_values = [0.3, 0.6]  # 不同的噪声标准差
for tau in tau_values:
    for m in m_values:
        for sigma in sigma_values:
            dataset = generate_nts_dataset(tau, m, sigma)
            datasets.append(dataset)

# 将列表转换为 DataFrame
df = pd.DataFrame(datasets, index=['date','x1','x2','x3','x4','x5','x6','x7','x8'])

# 转置 DataFrame
transposed_df = df.T

# 保存为 CSV 文件
csv_file_path = './data_sets/Sawtooth.csv'
transposed_df.to_csv(csv_file_path, index=False, header=True)

print(f"数据已保存到 {csv_file_path}")