import torch
import torch.nn as nn

# 模型输出，形状为 (32, 1)
val_output = torch.randn(32, 1)

# 实际标签，形状为 (32, 1)
label = torch.randn(32, 1)

# 试图计算损失时
# 下面的代码会报错
loss_fn = nn.MSELoss()
loss = loss_fn(val_output, label)  # 这将导致维度不匹配错误

