import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, head_num: int, dropout_rate: float) -> None:
        """
        多头注意力机制的实现

        params:
        d_model: int, 输入的维度，也就是每个token的embedding维度
        head_num: int, 注意力的头数
        dropout_rate: float, 随机丢失的概率

        return:
        None
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 输入的特征维度
        self.h = head_num        # 注意力头的数量

        # 每一头注意力的输出维度，d_k=d_v=d_model/h
        if d_model % head_num != 0 :
            raise ValueError('The values of d_model and head_num do not match. \
                             Please re-enter the values and make sure that d_model / head_num is an integer.')
        else:
            self.d = d_model // head_num

        # 定义线性变换层，生成Q、K、V的权重矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 定义dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 最后输出的线性层
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        """
        计算注意力得分

        params:
        q: torch.Tensor, 查询矩阵
        k: torch.Tensor, 键矩阵
        v: torch.Tensor, 值矩阵
        mask: torch.Tensor, 用于掩蔽的矩阵

        return:
        output: torch.Tensor, 计算后的输出
        """
        # 计算注意力得分，使用缩放点积注意力公式
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d)

        # 如果提供了掩蔽，则将无效位置的得分设为负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算softmax，得到注意力权重
        scores = F.softmax(scores, dim=-1)

        # 应用dropout
        scores = self.dropout(scores)

        # 使用注意力权重加权值矩阵
        output = torch.matmul(scores, v)

        return output, scores # 返回输出和注意力权重

    def forward(self, q: torch.Tensor, k: torch.Tensor,v: torch.Tensor,mask=None) -> torch.Tensor:
        """
        前向传播

        params:
        x: torch.Tensor, 输入的特征矩阵
        mask: torch.Tensor, 用于掩蔽的矩阵

        return:
        output: torch.Tensor, 经过注意力机制处理后的输出
        """
        batch_size = q.size(0)  # 获取批量大小

        # 通过线性变换生成Q、K、V，并调整形状以便多头注意力计算
        q = self.W_Q(q).view(batch_size, -1, self.h, self.d).transpose(1, 2)  # (batch_size, seq_len,d)->(batch_size,seq_len,head,d)->(batch_size, head_num, seq_len, d)
        k = self.W_K(k).view(batch_size, -1, self.h, self.d).transpose(1, 2)  # (batch_size, seq_len,d)->(batch_size,seq_len,head,d)->(batch_size, head_num, seq_len, d)
        v = self.W_V(v).view(batch_size, -1, self.h, self.d).transpose(1, 2)  # (batch_size, seq_len,d)->(batch_size,seq_len,head,d)->(batch_size, head_num, seq_len, d)

        # 计算注意力输出
        scores, attention_weights = self.attention(q, k, v, mask)

        # 将输出转换回原始形状，并通过最终线性层
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.out(output)

        return output, attention_weights # 返回输出和注意力权重

    def visualize_attention(self, attention_weights: torch.Tensor, head_index: int, seq_names: list):
        """
        可视化给定头的注意力权重
        params:
        attention_weights: torch.Tensor, 注意力权重
        head_index: int, 要可视化的头的索引
        seq_names: list, 序列中每个token的名称
        """
        attn = attention_weights[0][head_index].detach().cpu().numpy()  # 取第一个样本的注意力权重
        plt.figure(figsize=(16, 8))
        sns.heatmap(attn, xticklabels=seq_names, yticklabels=seq_names, cmap='viridis', cbar=True)
        plt.title(f'Attention Weights for Head {head_index}')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.show()

if __name__=='__main__':
    # 示例用法
    model = MultiHeadAttention(d_model=512, head_num=8, dropout_rate=0.1)
    input_tensor = torch.randn(32, 20, 512)  # (batch_size, seq_len, d_model)
    output, attention_weights = model(input_tensor)

    # 可视化第一个头的注意力权重
    seq_names = [f'Token {i}' for i in range(20)]  # 序列中的token名称
    model.visualize_attention(attention_weights, head_index=0, seq_names=seq_names)
