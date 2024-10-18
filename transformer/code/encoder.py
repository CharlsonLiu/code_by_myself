import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_norm import LayerNorm
from Position_Embeddding import TransformerEmbedding
from mutltihead_attention import MultiHeadAttention

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_rate: float) -> None:
        """
        初始化位置编码前馈网络。
        
        参数:
        d_model (int): 输入和输出的维度大小，通常与模型的embedding维度一致。
        hidden_dim (int): 前馈网络中间的隐藏层维度大小。
        dropout_rate (float): 随机失活的比例，用于防止过拟合。
        """
        super(PositionWiseFeedForward, self).__init__()
        
        
        # 随机失活层，用于在训练时对神经元进行随机失活以防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # 定义前馈网络中的第一个全连接层，输入维度为 d_model，输出维度为 hidden_dim
        self.L1 = nn.Linear(d_model, hidden_dim)
        
        # 定义前馈网络中的第二个全连接层，输入维度为 hidden_dim，输出维度为 d_model
        self.L2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，定义数据流的处理流程。
        
        参数:
        x (torch.Tensor): 输入张量，通常是来自前一层的输出，形状为 (batch_size, seq_len, d_model)。
        
        返回:
        torch.Tensor: 输出张量，形状与输入相同。
        """
        # 应用第一个全连接层，将输入从 d_model 映射到 hidden_dim
        x = self.L1(x)
        
        # 使用ReLU激活函数进行非线性变换
        x = F.relu(x)
        
        # 应用 dropout 进行随机失活，防止过拟合
        x = self.dropout(x)
        
        # 应用第二个全连接层，将 hidden_dim 映射回 d_model
        x = self.L2(x)
        
        # 返回处理后的输出
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, head_num: int, dropout_rate: float) -> None:
        """
        初始化位置编码前馈网络。
        
        参数:
        d_model (int): 输入和输出的维度大小，通常与模型的embedding维度一致。
        hidden_dim (int): 前馈网络中间的隐藏层维度大小。
        head_num (int): 注意力头的数量，传入以保持与外部一致性。
        dropout_rate (float): 随机失活的比例，用于防止过拟合。
        """
        super(EncoderLayer,self).__init__()

        self.Attention = MultiHeadAttention(d_model=d_model,head_num=head_num,dropout_rate=dropout_rate)
        self.FeedForward = PositionWiseFeedForward(d_model=d_model,hidden_dim=hidden_dim,dropout_rate=dropout_rate)

        self.LayerNorm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.LayerNorm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self,x:torch.Tensor,mask=None) -> torch.Tensor:
        _x = x
        x, _ = self.Attention(x,x,x,mask)  # 只接收注意力输出
        x = self.dropout1(x)
        x = self.LayerNorm1(x + _x)

        _x = x
        x = self.FeedForward(x)
        x = self.dropout2(x)
        x = self.LayerNorm2(x + _x)
        return x

        
class Encoder(nn.Module):
    def __init__(self,d_model: int, hidden_dim: int, head_num: int, dropout_rate: float,layer_num:int,\
                 vocab_size:int,max_len:int,device) -> None:
        """
        初始化Encoder类，该类构成了Transformer模型的编码器部分。
        
        参数:
        d_model (int): 输入embedding和编码器层的维度。
        hidden_dim (int): 前馈网络的隐藏层维度。
        head_num (int): 多头自注意力机制中的注意力头数量。
        dropout_rate (float): 随机失活的比例，用于防止过拟合。
        layer_num (int): 编码器层的数量。
        vocab_size (int): 词汇表的大小，用于词嵌入。
        max_len (int): 输入序列的最大长度。
        device (torch.device): 用于计算的设备（CPU或GPU）。
        """
        super().__init__()

        # Transformer embedding层，负责将输入的词汇转换为向量表示，同时添加位置编码
        self.embedding = TransformerEmbedding(max_len=max_len,d_model=d_model,device=device,\
                                              vocab_size=vocab_size,dropout_rate=dropout_rate)

        # 使用 nn.Sequential 创建多层的编码器层,会报错，只支持传递单一参数
        # python 中*[]代表解包，将一系列元素分开
        # self.layers = nn.Sequential(
        #     *[EncoderLayer(d_model=d_model, hidden_dim=hidden_dim, head_num=head_num, dropout_rate=dropout_rate)
        #       for _ in range(layer_num)]  # layer_num 表示编码器层的数量
        # )
        # 使用 ModuleList 而不是 Sequential 来存储多层编码器层
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, hidden_dim=hidden_dim, head_num=head_num, dropout_rate=dropout_rate)
             for _ in range(layer_num)]
        )

    def forward(self,x:torch.Tensor,s_mask=None)-> torch.Tensor:
        """
        前向传播函数。
        
        参数:
        x (torch.Tensor): 输入张量，通常为词汇索引，形状为 (batch_size, seq_len)。
        mask (torch.Tensor): 掩码张量，用于遮掩填充位置或未来位置，形状为 (batch_size, seq_len)。
        
        返回:
        torch.Tensor: 编码后的表示，形状为 (batch_size, seq_len, d_model)。
        """
        # 应用词嵌入和位置编码，将输入x转换为词向量
        x = self.embedding(x)
        # 手动遍历每一层的编码器层，并将 x 和 mask 作为参数传递给每一层
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
    

# 测试用例
def test_encoder():
    # 假设的参数配置
    d_model = 512         # Embedding 和编码器层的维度
    hidden_dim = 2048     # 前馈网络的隐藏层维度
    head_num = 8          # 多头注意力机制的头数
    dropout_rate = 0.1    # dropout比例
    layer_num = 6         # 编码器层的数量
    vocab_size = 10000    # 词汇表大小
    max_len = 100         # 输入序列的最大长度
    batch_size = 32       # 批大小
    seq_len = 50          # 输入序列的实际长度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测设备

    # 创建Encoder模型
    encoder = Encoder(
        d_model=d_model,
        hidden_dim=hidden_dim,
        head_num=head_num,
        dropout_rate=dropout_rate,
        layer_num=layer_num,
        vocab_size=vocab_size,
        max_len=max_len,
        device=device
    ).to(device)  # 将模型加载到设备上

    # 创建随机输入张量，模拟词汇索引 (batch_size, seq_len)
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # 前向传播，测试编码器的输出
    output = encoder(input_tensor)

    # 打印输出的形状，应该为 (batch_size, seq_len, d_model)
    print("Encoder Output Shape:", output.shape)

    # 断言输出形状是否正确
    assert output.shape == (batch_size, seq_len, d_model), "输出形状不正确"

# 执行测试
if __name__ == "__main__":
    test_encoder()
