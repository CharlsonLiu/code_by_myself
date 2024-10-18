import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-12):
        """
        自定义的 Layer Normalization 模块

        参数:
        - d_model: 输入张量的最后一个维度的大小，通常是特征维数。
        - eps: 数值稳定项，用来防止标准差为 0，避免除零错误，通常设置为一个非常小的值（如 1e-12）。
        """
        super(LayerNorm, self).__init__()

        # 可学习的缩放参数 gamma，初始化为全1，形状为 (d_model,)
        # 每个特征维度都会有一个独立的缩放参数。
        self.gamma = nn.Parameter(torch.ones(d_model))

        # 可学习的偏置参数 bias，初始化为全0，形状为 (d_model,)
        # 每个特征维度都会有一个独立的偏置参数。
        self.bias = nn.Parameter(torch.zeros(d_model))

        # eps 是防止计算过程中除零的极小值，避免标准差为 0 的情况
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，进行 Layer Normalization 操作。

        参数:
        - x: 输入张量，形状为 (batch_size, seq_len, d_model) 或 (batch_size, d_model)，
             其中最后一个维度是特征维度 d_model。
        
        返回:
        - 输出张量，形状与输入相同，特征维度被标准化并重新缩放和偏移。
        """
        # 计算输入张量在最后一个维度的均值 (mean)，
        # keepdim=True 保留维度信息，使得结果可以在后续操作中广播。
        # mean 的形状为 (batch_size, seq_len, 1) 或 (batch_size, 1)
        mean = x.mean(dim=-1, keepdim=True)

        # 计算输入张量在最后一个维度的方差 (variance)，
        # unbiased=False 表示使用非无偏估计，因为这是在深度学习中常用的计算方式。
        # var 的形状和 mean 一致。
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # 计算归一化结果，将每个特征减去均值并除以标准差 (sqrt(var + eps))。
        # eps 确保数值稳定性，防止标准差为 0。
        # output 的形状与 x 相同，即 (batch_size, seq_len, d_model)
        output = (x - mean) / torch.sqrt(var + self.eps)

        # 进行逐元素缩放和偏移：
        # - self.gamma 是形状为 (d_model,) 的可学习缩放参数，它会被广播到与 output 形状匹配
        # - self.bias 是形状为 (d_model,) 的可学习偏置参数，也会被广播到与 output 形状匹配
        # 最终对归一化后的输出乘以 gamma 并加上 bias。
        output = self.gamma * output + self.bias

        # 返回标准化并重新缩放、偏移后的张量
        return output


if __name__ =='__main__':
    # 输入张量
    x = torch.randn(2, 3, 4)  # 示例输入，形状为 (batch_size, seq_len, d_model)

    # 初始化自定义的 LayerNorm
    custom_layernorm = LayerNorm(d_model=4)

    # 使用 PyTorch 官方的 LayerNorm 进行对比
    official_layernorm = nn.LayerNorm(normalized_shape=4, eps=1e-12)

    # 为了对比公平，将 PyTorch 官方的 gamma 和 bias 与自定义模块的参数一致化
    with torch.no_grad():
        official_layernorm.weight.copy_(custom_layernorm.gamma)
        official_layernorm.bias.copy_(custom_layernorm.bias)

    # 获取自定义 LayerNorm 的输出
    custom_output = custom_layernorm(x)

    # 获取官方 LayerNorm 的输出
    official_output = official_layernorm(x)

    # 检查两个输出是否相等
    are_equal = torch.allclose(custom_output, official_output, atol=1e-6)

    # 打印结果
    print("自定义 LayerNorm 输出:\n", custom_output)
    print("官方 LayerNorm 输出:\n", official_output)
    print("两者是否一致:", are_equal)
