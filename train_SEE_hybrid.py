import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from Model_hybridSEE import HybridModel
from bayes_opt import BayesianOptimization

# 设置字体和随机种子
plt.rcParams['font.family'] = 'SimHei'
torch.manual_seed(42)
n_qubits = 8

def partial_trace(rho, dims, trace_out):
    n = len(dims)
    rho = rho.reshape(dims + dims)
    axes = list(range(n))
    keep_axes = [i for i in range(n) if i not in trace_out]
    trace_axes = trace_out
    
    new_order = keep_axes + [n + i for i in keep_axes] + trace_axes + [n + i for i in trace_axes]
    rho = rho.permute(new_order)
    
    keep_dims = torch.prod(torch.tensor([dims[i] for i in keep_axes])).item()
    trace_dims = torch.prod(torch.tensor([dims[i] for i in trace_axes])).item()
    rho = rho.reshape(keep_dims, trace_dims, keep_dims, trace_dims)
    
    rho_traced = rho.sum(dim=(1, 3))
    rho_traced = (rho_traced + rho_traced.conj().T) / 2
    trace = torch.trace(rho_traced)
    if abs(trace) > 1e-10:
        rho_traced = rho_traced / trace
    else:
        print(f"Warning: rho_traced trace={trace}, returning identity")
        return torch.eye(keep_dims, dtype=rho.dtype) / keep_dims
    return rho_traced

def compute_see(state, n_qubits):
    expected_dim = 2 ** n_qubits
    if state.shape != (expected_dim,):
        raise ValueError(f"state shape {state.shape} does not match expected ({expected_dim},)")
    if torch.norm(state) < 1e-10 or torch.any(torch.isnan(state)):
        print(f"Warning: Invalid state, norm={torch.norm(state)}")
        return np.zeros(n_qubits)
    
    sees = []
    state = state / torch.norm(state + 1e-10)
    rho = torch.outer(state.conj(), state)
    print("Density matrix rho:\n", rho)
    
    dims = [2] * n_qubits
    for i in range(n_qubits):
        rho_i = partial_trace(rho, dims, [i])
        print(f"Qubit {i} reduced density matrix rho_i:\n", rho_i)
        eigenvalues = torch.linalg.eigvalsh(rho_i)
        print(f"Qubit {i} eigenvalues:\n", eigenvalues)
        eigenvalues = torch.clamp(eigenvalues, min=0.0, max=1.0)
        eigenvalues_sum = torch.sum(eigenvalues)
        print(f"Qubit {i} eigenvalues sum:", eigenvalues_sum)
        if eigenvalues_sum > 1e-10:
            eigenvalues = eigenvalues / eigenvalues_sum
        else:
            sees.append(0.0)
            continue
        print(f"Qubit {i} normalized eigenvalues:\n", eigenvalues)
        see = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-12))
        print(f"Qubit {i} SEE before clamp:", see)
        see = torch.clamp(see, min=0.0, max=1.0)
        sees.append(see.item())
    
    return np.array(sees)

# 数据加载和预处理
current_path = os.getcwd()
file_path = os.path.join(current_path, 'data_sets', 'MackeyGlass.csv')
save_path = os.path.join(current_path, 'data_sets', 'lstm_model.pth')
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
features = data[['x1','x2','x3','x4','x5','x6','x7','x8']].values

train_size1 = int(len(features) * 0.6)
train_size2 = int(len(features) * 0.8)
train_features = features[:train_size1]
val_features = features[train_size1:train_size2]
test_features = features[train_size2:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_features)
scaled_val = scaler.transform(val_features)
scaled_test = scaler.transform(test_features)

train_data = scaled_train
val_data = scaled_val
test_data = scaled_test

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        b = data[i + time_step].reshape(1, 8)
        y.append(b)
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train, y_train)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 贝叶斯优化
def bayesian_objective(alpha):
    alpha = max(0.0, min(1.0, alpha))
    model = HybridModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output1, output2, _ = model(batch_X)
            loss1 = criterion(output1, batch_y)
            loss2 = criterion(output2, batch_y)
            loss = alpha * loss1 + (1 - alpha) * loss2
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs, _, _ = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    return -val_loss.item()

pbounds = {'alpha': (0.8, 1)}
optimizer = BayesianOptimization(f=bayesian_objective, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=10)

best_alpha = optimizer.max['params']['alpha']
print(f'Best alpha: {best_alpha}')

# 训练模型
alpha = best_alpha
model = HybridModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output1, output2, _ = model(batch_X)
        loss1 = criterion(output1, batch_y)
        loss2 = criterion(output2, batch_y)
        loss = alpha * loss1 + (1 - alpha) * loss2
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), save_path)
print("模型已保存为 'lstm_model.pth'")

# 测试和SEE分析
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
test_loss = 0
predictions = []
actuals = []
sees_all = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        output1, output2, quantum_state = model(batch_X)
        loss1 = criterion(output1, batch_y)
        loss2 = criterion(output2, batch_y)
        loss = alpha * loss1 + (1 - alpha) * loss2
        outputs = output1
        test_loss += loss.item()
        
        predictions.append(outputs.numpy())
        actuals.append(batch_y.numpy())
        
        # 计算SEE
        for state in quantum_state:
            sees = compute_see(state, n_qubits)
            sees_all.append(sees)

# 计算平均测试损失
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# 合并批次结果
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)
sees_all = np.array(sees_all)

# 反归一化
def inverse_scale_predictions(scaler, preds, actuals, time_step):
    dummy_data = np.zeros((len(preds) + time_step, scaler.n_features_in_))
    dummy_data[time_step:] = preds.reshape(-1, scaler.n_features_in_)
    inv_preds = scaler.inverse_transform(dummy_data)[time_step:]
    dummy_data[time_step:] = actuals.reshape(-1, scaler.n_features_in_)
    inv_actuals = scaler.inverse_transform(dummy_data)[time_step:]
    return inv_preds, inv_actuals

inv_preds, inv_actuals = inverse_scale_predictions(scaler, predictions, actuals, time_step)
target_idx = 0
predicted_target = inv_preds[:, target_idx]
actual_target = inv_actuals[:, target_idx]

# 预测结果可视化
plt.figure(figsize=(12, 6))
plt.plot(actual_target, label='Actual Values')
plt.plot(predicted_target, label='Predicted Values')
plt.title('Test Set Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('x1')
plt.legend()
plt.show()

# SEE Visualization
plt.figure(figsize=(12, 6))
for i in range(n_qubits):
    plt.plot(sees_all[:, i], label=f'Qubit {i}')
plt.title('Single-Site Entanglement Entropy (SEE) for Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Entanglement Entropy (bits)')
plt.legend()
plt.show()

# Average SEE
avg_sees = np.mean(sees_all, axis=0)
plt.figure(figsize=(8, 5))
plt.bar(range(n_qubits), avg_sees)
plt.title('Average Single-Site Entanglement Entropy per Qubit')
plt.xlabel('Qubit Index')
plt.ylabel('Average Entanglement Entropy (bits)')
plt.show()

# 保存差值
difference = inv_actuals - inv_preds
difference_df = pd.DataFrame(difference, columns=['x1','x2','x3','x4','x5','x6','x7','x8'])
difference_df.to_csv(f'difference_hybrid_{num_epochs}.csv', index=False)
