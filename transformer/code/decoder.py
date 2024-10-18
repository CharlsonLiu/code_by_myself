import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layer_norm import LayerNorm
from Position_Embeddding import TransformerEmbedding
from mutltihead_attention import MultiHeadAttention
from encoder import PositionWiseFeedForward, EncoderLayer

class DecoderLayer(nn.Module):
    def __init__(self,d_model:int,head_num:int,hidden_dim:int,dropout_rate:float) -> None:
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model,head_num=head_num,
                                             dropout_rate=dropout_rate)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.crossAtten = MultiHeadAttention(d_model=d_model,head_num=head_num,
                                             dropout_rate=dropout_rate)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.Linear = PositionWiseFeedForward(d_model=d_model,hidden_dim=hidden_dim,
                                              dropout_rate=dropout_rate)
        
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self,decoder:torch.Tensor,encoder:torch.Tensor,t_mask:torch.Tensor,s_mask:torch.Tensor)->torch.Tensor:
        """
        params:
        decoder:解码器的输出
        encoder：编码器的输出
        t_mask：用于自回归机制，确保解码器在每一步生成词之前只能看到之前的词。
                是一个上三角矩阵
        s_mask:对encoder的输出做masking，防止模型在处理变长序列时受padding位置的影响。
        """
        _x = decoder

        x = self.attention(decoder,decoder,decoder,t_mask)
        x = self.norm1(self.dropout1(x) + _x)
        _x = x

        x = self.crossAtten(x,encoder,encoder,s_mask)
        x = self.norm2(self.dropout2(x) + _x)
        _x = x

        x = self.Linear(x)
        x = self.norm3(self.dropout3(x) + _x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,dec_voc_size:int,max_len:int,d_model:int,hidden_dim:int,
                 n_head:int,layer_num:int,dropout_rate:float,device)->torch.Tensor:
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=dec_voc_size,d_model=d_model,
                                              device=device,max_len=max_len,dropout_rate=dropout_rate)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model,head_num=n_head,hidden_dim=hidden_dim,dropout_rate=dropout_rate)

            ]
        )
        self.linear = nn.Linear(d_model,dec_voc_size)

    def forward(self,decoder,encoder,t_mask,s_mask):
        decoder = self.embedding(encoder)
        for layer in self.layers:
            decoder = layer(decoder,encoder,t_mask,s_mask)
        decoder = self.linear(decoder)
        return decoder