import torch
import torch.nn.functional as F

# 生成随机输入数据
input_tensor = torch.randn(16,676, 4 )

# 创建卷积核 (修改形状)
kernel_size = 1
in_channels = 676  # 必须与输入张量的通道数一致
out_channels = 128
weight = torch.randn(out_channels, in_channels, kernel_size)

# 执行一维卷积
output = F.conv1d(input_tensor, weight, stride=1, padding=0)

print("输出张量形状:", output.shape)