import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        初始化 TokenEmbedding 类。

        参数:
        - vocab_size (int): 词汇表大小，即嵌入矩阵的行数。每个不同的 token（单词或子词）都对应一个唯一的嵌入向量。
        - d_model (int): 嵌入向量的维度，即嵌入矩阵的列数。每个 token 将被映射为一个 d_model 维度的向量。

        此构造函数会自动创建一个嵌入矩阵，该矩阵将 token 索引映射到其对应的嵌入向量。嵌入向量用于将离散的 token 转换为连续的高维空间表示，便于模型进行计算。

        返回:
        - None: 此构造函数不返回任何值，但会初始化一个嵌入矩阵，形状为 (vocab_size, d_model)。

        注意:
        - padding_idx=1 表示索引为1的位置被视为填充位置，其嵌入值为零向量，且在梯度更新中不参与计算。这对于处理变长序列非常有用，以避免填充位置影响模型训练。
        """
        # 调用父类 nn.Embedding 的构造函数，初始化嵌入矩阵
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)



# 定义positional embedding类
class PositionalEmbedding(nn.Module):
    def __init__(self,max_len:int,d_model:int,device) -> torch.Tensor :
        
        super(PositionalEmbedding,self).__init__()
        # 初始化encoding矩阵，注意不需要梯度，因为这个位置不会随着模型训练而更新
        self.encoding = torch.zeros(size=(max_len,d_model),device=device,requires_grad=False)

        # 在dim=1上面做维度添加是因为每个token的embedding占据一行，
        # 自然token的位置排列按照行排序，形成 [max_len, 1] 形状
        pos = torch.arange(0, max_len, device=device, dtype=torch.float).unsqueeze(dim=1)
        
        # 生成一个步长为2的张量_2i，表示位置编码中使用到的指数的偶数维度
        # step=2 是因为位置编码公式中只用到了 2i 维度
        _2i = torch.arange(0, d_model, step=2, device=device, dtype=torch.float)

        # 计算偶数维度的位置编码，公式为 sin(pos / 10000^(2i/d_model))
        # pos 是形状为 [max_len, 1]，_2i 是形状为 [d_model/2]，
        # 通过广播机制扩展为 [max_len, d_model] 形状
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))

        # 计算奇数维度的位置编码，公式为 cos(pos / 10000^(2i/d_model))
        # 与上面相同，pos 和 _2i 通过广播扩展形状，结果存储在奇数维度列
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

        # 保持embedding层不参与梯度更新（即不训练）
        self.encoding = self.encoding.detach()
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x:[batch,token_length],x包含的是批次大小以及token数量
        # token数量也就代表着token的索引
        return self.encoding[:x.size(1),:]
    
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int,device,max_len:int,dropout_rate:float):
        super(TransformerEmbedding,self).__init__()
        self.token_emb = TokenEmbedding(vocab_size,d_model)
        self.pos_emb = PositionalEmbedding(max_len,d_model,device)
        self.dropout = nn.Dropout(p = dropout_rate)
    
    def forward(self,x:torch.Tensor):
        # x:[batch_size,token_length]
        token_embed = self.token_emb(x)
        pos_embed = self.pos_emb(x)

        return self.dropout(token_embed+pos_embed)
    

def visualize_positional_embedding(pos_emb: PositionalEmbedding):
    # 将位置编码转换为 NumPy 数组以进行可视化
    encoding = pos_emb.encoding.cpu().numpy()
    
    # 可视化位置编码
    plt.figure(figsize=(10, 6))
    plt.imshow(encoding, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Positional Embedding Visualization')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Positions')
    plt.show()


if __name__ == "__main__":
    # 示例参数
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 嵌入维度
    max_len = 50       # 位置编码最大长度
    dropout_rate = 0.1  # Dropout 概率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'      # 设备选择，CPU 或 GPU

    # 创建 PositionalEmbedding 实例
    pos_emb = PositionalEmbedding(max_len, d_model, device)

    # 可视化位置编码
    visualize_positional_embedding(pos_emb)
