import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layer_norm import LayerNorm
from Position_Embeddding import TransformerEmbedding
from mutltihead_attention import MultiHeadAttention
from encoder import Encoder
from decoder import Decoder


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

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q, len_k = q.size(1),k.size(1)
        q = q.ne(*pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1,1,1,len_k)

        k = k.ne(*pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1,1,len_q,1)

        mask = q&k

        return mask
    
    def causal_mask(self,q,k):
        mask = torch.trill(torch.ones(q.size(1),k.size(1))).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self,src,trg):
        src_mask = self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx) * self.causal_mask(trg,trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(src,trg,trg_mask,src_mask)
        return out