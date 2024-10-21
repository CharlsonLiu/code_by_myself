import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layer_norm import LayerNorm
from Position_Embeddding import TransformerEmbedding
from mutltihead_attention import MultiHeadAttention
from encoder import Encoder
from decoder import Decoder

"""
解码器（Decoder）数据流和交叉注意力机制说明

在 Transformer 模型的解码器中，数据流主要有两个部分：

1. 编码器输出（Encoder Output）：
   - 经过编码器处理后的源文本表示，包含源文本的上下文信息。
   - 在交叉注意力计算中，编码器输出作为键（K）和值（V）。

2. 目标文本（Target Text）：
   - 训练时的目标序列（标签），经过嵌入处理（embedding）并添加位置编码。
   - 目标文本在输入解码器之前会经过掩码自注意力处理，以确保模型在生成当前词时仅依赖于先前的词（自回归特性）。

### 数据流处理步骤：

1. **目标文本的嵌入**：
   - 目标文本首先通过嵌入层（embedding layer）转换为密集向量表示。

2. **自注意力（Masked Self-Attention）**：
   - 嵌入后的目标文本序列会被送入自注意力机制，使用因果掩码（causal mask）来确保生成过程中只关注当前词之前的词。

3. **交叉注意力（Cross-Attention）**：
   - 自注意力的输出作为查询（Q），编码器的输出作为键（K）和值（V）。
   - 交叉注意力使解码器能够参考编码器提供的源信息，从而生成更加上下文相关的目标文本。

### 掩码的作用：

- **因果掩码（Causal Mask）**：
  - 确保解码器在每一步生成时仅依赖于当前词之前的词，防止信息泄露。
  - 形式上，这是一种上三角矩阵，标记了哪些词可以被关注。

- **填充掩码（Padding Mask）**：
  - 在处理变长序列时，填充掩码用于避免模型在注意力计算中受到填充位置的影响。
  - 它通过记录填充位置的索引来创建布尔掩码，以屏蔽无效的填充元素。

### 数据流示例：
- 输入到解码器的内容：
  - `encoder_output`：编码器的输出（源文本的上下文表示）。
  - `trg`：目标文本（经过嵌入的序列）。

注意：
- 编码器输出作为交叉注意力的 K 和 V，目标文本的嵌入作为 Q。
- 此结构使得解码器能够有效结合源文本信息与目标文本上下文，以更好地完成序列到序列的任务。
"""

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,trg_pad_idx,
                 enc_voc_size:int,dec_voc_size:int,
                 d_model:int,hidden_dim:int,
                 n_head:int,dropout_rate:float,
                 max_len:int,num_layers:int,
                 device):
        super(Transformer,self).__init__()
        self.encoder = Encoder(d_model=d_model,
                               hidden_dim=hidden_dim,
                               head_num=n_head,
                               dropout_rate=dropout_rate,
                               layer_num=num_layers,
                               vocab_size=enc_voc_size,
                               max_len=max_len,
                               device=device
                               )
        self.decoder = Decoder(d_model=d_model,
                               dec_voc_size=dec_voc_size,
                               max_len=max_len,
                               hidden_dim=hidden_dim,
                               n_head=n_head,
                               layer_num=LayerNorm,
                               dropout_rate=dropout_rate,
                               device=device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        # q 的形状为 (batch_size, len_q)， k 的形状为 (batch_size, len_k) 
        # 注意，注意力中QK.T->(batchsize,n_head,len_q,len_k)
        # 获取序列的长度，也就是有多少个token
        len_q, len_k = q.size(1), k.size(1)

        # 使用 ne 方法生成一个布尔类型的张量，判断 q 中哪些位置不是填充值
        # q.ne(*pad_idx_q) 返回一个与 q 形状相同的布尔张量
        # True 表示对应位置不是填充，False 表示是填充
        q = q.ne(*pad_idx_q).unsqueeze(1).unsqueeze(3)
        # q 的形状现在为 (batch_size, 1, len_q, 1)，用于后续广播操作

        # 重复 q 张量，使其形状变为 (batch_size, 1, len_q, len_k)
        # 这样可以在后续的按位与操作中与 k 的掩码匹配
        q = q.repeat(1, 1, 1, len_k)

        # 对 k 执行相同的操作，生成一个布尔类型的张量
        k = k.ne(*pad_idx_k).unsqueeze(1).unsqueeze(2)
        # k 的形状现在为 (batch_size, 1, 1, len_k)

        # 重复 k 张量，使其形状变为 (batch_size, 1, len_q, len_k)
        k = k.repeat(1, 1, len_q, 1)

        # 通过按位与操作生成最终的掩码
        # 只有在 q 和 k 中对应位置都为 True 的情况下，mask 的值才为 True
        mask = q & k

        # 返回最终的掩码，形状为 (batch_size, len_q, len_k)
        return mask

    
    def causal_mask(self, q, k):
        # 创建一个上三角矩阵，形状为 (len_q, len_k)
        # 上三角矩阵的元素为 1，表示可以访问的位置，0 表示不可访问的位置
        # q.size(1) 表示查询序列的长度，k.size(1) 表示键序列的长度
        mask = torch.triu(torch.ones(q.size(1), k.size(1))).type(torch.BoolTensor).to(self.device)

        # 返回生成的因果掩码
        return mask


    def forward(self,src,trg):
        # 生成 padding mask
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        
        # 生成 causal mask，防止未来信息泄露
        trg_causal_mask = self.causal_mask(trg, trg)
        
        # 合并 mask，通常用 "&" 而不是 "*"
        trg_mask = trg_pad_mask & trg_causal_mask
        
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg,enc,trg_mask,src_mask)
        return out