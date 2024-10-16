import torch
import torch.nn as nn

# 输入参数
batch_size = 32  # 批量大小
seq_len = 10     # 序列长度
d_model = 64     # 输入特征维度
d_k = d_v = 64   # Q, K, V 的特征维度

# 模拟输入数据 (batch_size, seq_len, d_model)
X = torch.randn(batch_size, seq_len, d_model)

# 定义线性变换用于生成 Q, K, V
W_Q = nn.Linear(d_model, d_k)
W_K = nn.Linear(d_model, d_k)
W_V = nn.Linear(d_model, d_v)

# 生成 Q, K, V 矩阵 (batch_size, seq_len, d_k/d_v)
Q = W_Q(X)  # (batch_size, seq_len, d_k)
K = W_K(X)  # (batch_size, seq_len, d_k)
V = W_V(X)  # (batch_size, seq_len, d_v)
print(Q.shape)
# 计算注意力权重
# 转置 K 矩阵以便点积计算 (batch_size, d_k, seq_len)
K_transpose = K.transpose(-1, -2)  # 交换最后两维

# QK^T / sqrt(d_k) 计算 (batch_size, seq_len, seq_len)
attention_scores = torch.matmul(Q, K_transpose) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# 通过 softmax 获得注意力权重 (batch_size, seq_len, seq_len)
attention_weights = nn.Softmax(dim=-1)(attention_scores)

# 计算加权的 V (batch_size, seq_len, d_v)
attention_output = torch.matmul(attention_weights, V)

# 输出的 attention_output 形状为 (batch_size, seq_len, d_v)
print(attention_output.shape)

print(torch.ones(size=(10,8,16)).view(10,-1,4).shape)
