import torch
import torch.nn as nn
import tensorcircuit as tc
import logging
# 禁用tensorcircuit的冗余日志
tc.set_backend("pytorch")
logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.WARNING)  # 将日志级别设为WARNING及以上

n_qubits = 8

def quantum_circuit(inputs, params):
    c = tc.Circuit(n_qubits)
    # 输入参数处理（适配批量输入）
    # inputs 的形状为 (time_steps, 16)
    # params 的形状为定义的weights_shape
    # print(f'params shape is {params.shape}')
    # print(f'inputs shape is {inputs.shape}')
    N = len(inputs)
    
    for i in range(n_qubits):
        # 初始 Hadamard 门
        c.h(i)
       
    # for j in range(N):  
    #     # 批量参数化旋转
    #     for i in range(n_qubits):
    #         c.ry(i, theta=params[j * 24 + i])  
    #     for i in range(n_qubits):
    #         c.rz(i, theta=params[j * 24 + i + n_qubits])
    #     for i in range(n_qubits):
    #         c.ry(i, theta=params[j * 24 + i + 2*n_qubits])
         
    for j in range(N):  
        # 批量参数化旋转
        for i in range(n_qubits):
            c.ry(i, theta=inputs[j][i])  
        for i in range(n_qubits):
            c.rz(i, theta=inputs[j][i + n_qubits])
        for i in range(n_qubits):
            c.ry(i, theta=inputs[j][i + 2*n_qubits])
        
        # CNOT 层（保持批量维度）
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
            c.rz(i + 1, theta=inputs[j][i + 3*n_qubits])  
            c.cnot(i, i + 1)
        
        # 环状连接
        c.cnot(n_qubits - 1, 0)
        c.rz(0, theta=inputs[j][4*n_qubits-1])  
        c.cnot(n_qubits - 1, 0)

            
    # 批量测量
    expectations = [tc.backend.real(c.expectation_ps(z=[i])) for i in range(8)]
    return tc.backend.stack(expectations, axis=-1)  # 形状为 (batch_size, 7)

class QuanvLayer(nn.Module):
    def __init__(self):
        super(QuanvLayer, self).__init__()
        self.qlayer = tc.TorchLayer(quantum_circuit, weights_shape=(24*10,)) #10为timestep

    def forward(self, x):
        # 确保输入形状为 (batch_size, 16)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 如果输入是 1 维的，增加一个批量维度
        return self.qlayer(x)

class HybridModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_classes=8):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_trad = nn.Linear(hidden_size, num_classes)
        self.fc_quan = nn.Linear(4, num_classes)  # 输出量子线路需要的 16 个参数
        self.quantum_circuit = QuanvLayer()

    def forward(self, x):
        # LSTM 处理
        out, _ = self.lstm(x)  # 输出形状: (batch_size, seq_len, hidden_size)
        # print(f'out shape is {out.shape}')
        # 传统路径：LSTM最后时间步 → 全连接
        out1 = self.fc_trad(out[:, -1, :])  # 形状: (batch_size, num_classes)
        out1 = out1.view(len(out),1,8)
        # print(f'out1 shape is {out1.shape}')
        # 量子路径：LSTM全时间步 → 生成量子参数 → 量子电路 → 全连接
        # quantum_output = self.quantum_circuit(out)  # 形状: (batch_size, 4)
        # print(f'quantum_output shape is {quantum_output.shape}')
        # quantum_output = quantum_output.view(len(out),1,7)
        
        # return quantum_output
        return out1