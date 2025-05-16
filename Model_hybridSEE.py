import torch
import torch.nn as nn
import tensorcircuit as tc
import numpy as np
import logging

# 禁用tensorcircuit的冗余日志
tc.set_backend("pytorch")
logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.WARNING)

n_qubits = 8

def quantum_circuit(inputs, params):
    c = tc.Circuit(n_qubits)
    N = len(inputs)
    
    for i in range(n_qubits):
        c.h(i)
            
    for j in range(N):  
        for i in range(n_qubits):
            c.ry(i, theta=inputs[j][i])
            c.rz(i, theta=inputs[j][i + n_qubits])
            c.ry(i, theta=inputs[j][i + 2 * n_qubits])
        
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
            c.rz(i + 1, theta=inputs[j][i + 3 * n_qubits])
            c.cnot(i, i + 1)
        
        c.cnot(n_qubits - 1, 0)
        c.rz(0, theta=inputs[j][4 * n_qubits - 1])
        c.cnot(n_qubits - 1, 0)
    
    # 返回期望值和量子态
    expectations = [tc.backend.real(c.expectation_ps(z=[i])) for i in range(n_qubits)]
    state = c.state()  # 获取量子态
    return tc.backend.stack(expectations, axis=-1), state

class QuanvLayer(nn.Module):
    def __init__(self):
        super(QuanvLayer, self).__init__()
        self.qlayer = tc.TorchLayer(quantum_circuit, weights_shape=(16,))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        expectations, state = self.qlayer(x)
        return expectations, state

class HybridModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_classes=8):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_trad = nn.Linear(hidden_size, num_classes)
        self.fc_quan = nn.Linear(4, num_classes)  # 调整为16以匹配量子电路输入
        self.quantum_circuit = QuanvLayer()

    def forward(self, x):
        out, _ = self.lstm(x)
        out1 = self.fc_trad(out[:, -1, :])
        out1 = out1.view(len(out), 1, 8)
        
        # 通过fc_quan生成量子电路参数
        quantum_output, quantum_state = self.quantum_circuit(out)
        
        return out1, quantum_output, quantum_state