#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :xDeepFm.py
@说明        :
@时间        :2024/11/18 16:16:21
@作者        :Liu Ziyue
'''

import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,TensorDataset
import torch.nn.init as init
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def process_data(data_df,dense_feature,sparse_feature):
    data_df[dense_feature] = data_df[dense_feature].fillna(0.0)
    for f in dense_feature:
        data_df[f] = data_df[f].apply(lambda x: np.log(x+1) if x > -1 else -1)
    
    data_df[sparse_feature] = data_df[sparse_feature].fillna('-1')
    for f in sparse_feature:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])
    return data_df[dense_feature + sparse_feature]


class CIN(nn.Module):
    def __init__(self,sparse_num, embed_dim, hidden_dim=[128,128]):
        super(CIN, self).__init__()
        '''
        :params
        - sparse_num -> int:the num of sparse features
        - embed_dim -> int: The dimension of the original embedding for each sparse feature.
        - hidden_dim -> List： A list of the hidden units num of each hidden layer
        '''
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.sparse_num = sparse_num
        # CIN 每一层的大小，以list的形式展现，更利于计算
        self.field_nums = [self.sparse_num] + hidden_dim

        # 过滤器
        # pytorch中的卷积权重是[out_channels, in_channels, kernel_size]
        # 当kernel_size=1时，相当于点卷积，也就是每个通道的值经过一个标量权重
        # 与偏置项计算后直接输出，没有空间上的感受野
        # 在这里的通道指的是输入特征的数量
        self.cin_W = nn.ParameterDict(
            {
                f'CIN_W_{i}':nn.Parameter(
                    torch.randn(self.field_nums[i + 1], self.field_nums[0] * self.field_nums[i], 1 )
                    )
                    for i in range(len(self.field_nums) - 1)
            }
        )
        # 用 Xavier 初始化
        for param in self.cin_W.values():
            nn.init.xavier_uniform_(param)


    def forward(self,input):
        # input :[batch, sparse_feature_num, embed_dim]
        hidden_layers_results = [input] # 这个存储当前层的结果

        # 从embedding的维度把张量一个个的切开,这个为了后面逐通道进行卷积
        # 这个结果是个list，list长度是embed_dim,
        # 每个元素维度是[None, field_nums[0], 1]  field_nums[0]即输入的特征个数
        # 即把输入的[None, field_num, embed_dim]，切成了embed_dim个[None, field_nums[0], 1]的张量
        # 最后的张量应该是[embed_dim, batch, feature_num, 1 for broadcast]
        split_x_0 = self.process_split_tensor(hidden_layers_results[0], self.embed_dim)

        for idx, size in enumerate(self.hidden_dim):
            # 这个操作和上面同理，也是为了逐通道卷积的时候更加方便，分割的是当前层的输入Xk-1
            # embed_dim个[None, field_nums[i], 1] 
            # feild_nums[i] 当前隐藏层单元数量
            split_x_k = self.process_split_tensor(hidden_layers_results[-1],self.embed_dim)

            # 外积运算
            # pytorch中，向量的外积就是[1,a] [b,1]之间的矩阵乘法
            # 因为pytorch会自动广播
            # 当然也可以用torch.outer
            # [embed_dim, None, field_nums[0], field_nums[i]]
            out_product_res_m = torch.matmul(split_x_0,split_x_k.transpose(-2, -1))
            # 后两维合并，方便运算，因为卷积核的权重矩阵是合并定义的
            out_product_res_o = out_product_res_m.view(self.embed_dim,-1,self.field_nums[0] * self.field_nums[idx])
            # [None, field_nums[0]*field_nums[i], dim]
            out_product_res = out_product_res_o.permute(1,2,0)

            # 卷积运算
            # 这个理解的时候每个样本相当于1张通道为1的照片 dim为宽度，
            # field_nums[0]*field_nums[i]为长度
            # 这时候的卷积核大小是field_nums[0]*field_nums[i]的, 
            # 这样一个卷积核的卷积操作相当于在dim上进行滑动，每一次滑动会得到一个数
            # 这样一个卷积核之后，会得到dim个数，即得到了[None, dim, 1]的张量， 
            # 这个即当前层某个神经元的输出
            # 当前层一共有field_nums[i+1]个神经元， 也就是field_nums[i+1]个卷积核，
            # 最终的这个输出维度[None, dim, field_nums[i+1]]
            # 选择某个卷积核进行卷积（例如，'CIN_W_0'）
            filter_weight = self.cin_W[f'CIN_W_{idx}']

            # 使用 F.conv1d 执行卷积，stride=1, padding=0 (等同于 VALID padding)
            cnn_layer_out = F.conv1d(out_product_res, filter_weight, stride=1, padding=0)
            hidden_layers_results.append(cnn_layer_out)

        # 最后CIN的结果取中间层的输出，不需要第0层
        final_result = hidden_layers_results[1:]
        # [None, H1+H2+...HT, dim]
        result = torch.concat(final_result,dim=1)
        # [None, H1+H2+..HT]
        result = torch.sum(result,dim=2,keepdim=False)

        return result

    def process_split_tensor(self,input_tensor, embed_dim):
        """
        对输入张量进行拆分并调整维度，最终返回形状为 [embed_dim, batch_size, field_nums[0], 1] 的张量。
        
        :param input_tensor: 输入张量，形状为 [batch_size, field_nums[0], total_embed_dim]
        :param embed_dim: 每个子张量的嵌入维度，即拆分的维度大小
        :return: 调整后的张量，形状为 [embed_dim, batch_size, field_nums[0], 1]
        """
        # 按 embed_dim 拆分张量
        split_tensor = torch.split(input_tensor, embed_dim, dim=2)

        # 处理拆分后的张量
        processed_splits = []

        for split in split_tensor:
            # 调整维度顺序为 [embed_dim, batch_size, field_nums[0]]
            split = split.permute(2, 0, 1)
            
            # 增加新的维度，变为 [embed_dim, batch_size, field_nums[0], 1]
            split = split.unsqueeze(-1)
            
            # 添加到列表
            processed_splits.append(split)

        # 将所有子张量堆叠在一起，最终形状为 [embed_dim, batch_size, field_nums[0], 1]
        final_tensor = torch.cat(processed_splits, dim=0)

        return final_tensor

class xDeepFM(nn.Module):
    def __init__(self, Linearfeature, DNNFeature, dense_dim,
                    sparse_dim, embed_dim):
        super(xDeepFM, self).__init__()

        # 线性层部分，这里和wide and deep是一样的
        # 稠密和稀疏特征分开处理，得到各自的logit，
        # 相加就是线性部分的logit
        self.linear_sparse_feat = Linearfeature['sparse']

        self.linear_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.linear_sparse_feat
        ])

        # 构建线性部分的稠密logit计算模块
        self.linear_Dlogit = nn.Linear(dense_dim, 1)

        # CIN部分，也是这篇论文的核心部分
        self.cin = CIN(sparse_num=sparse_dim,embed_dim=embed_dim)
        self.cin_logits = nn.Linear(256,1)

        # DNN部分
        self.dnn_sparse_feature = DNNFeature['sparse']

        self.dnn_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.dnn_sparse_feature
        ])
        
        self.dnn_layers = nn.Sequential(
            nn.Linear(dense_dim + sparse_dim * embed_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # 添加权重初始化
        self.init_weights()

    def forward(self, linear_dense_data, linear_sparse_data, dnn_dense_data, dnn_sparse_data):
        # 线性层
        linear_sparse = torch.stack([emb(linear_sparse_data[:, i]) for i, emb in enumerate(self.linear_embeddings)], dim=1)
        linear_sparse_logits = linear_sparse.sum(dim=1)
        linear_dense_logits = self.linear_Dlogit(linear_dense_data)
        linear_logit = linear_dense_logits + linear_sparse_logits
        
        # CIN部分，单独处理sparse部分，
        # 因此可以借用DNN部分的embedding
        cin_input = torch.stack([emb(dnn_sparse_data[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=1)
        exFM_out = self.cin(cin_input)
        exFM_logits = self.cin_logits(exFM_out)

        # DNN层
        dnn_sparse = torch.cat([emb(dnn_sparse_data[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=1)
        dnn_input = torch.cat([dnn_dense_data, dnn_sparse], dim=1)
        dnn_logit = self.dnn_layers(dnn_input)

        output_logit = linear_logit + dnn_logit + exFM_logits

        return output_logit

    def init_weights(self):
        # 对Embedding层进行初始化
        for emb in self.linear_embeddings:
            init.xavier_uniform_(emb.weight)  # Xavier初始化
        for emb in self.dnn_embeddings:
            init.xavier_uniform_(emb.weight)

        # 对线性层进行初始化
        for layer in self.dnn_layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier初始化
                init.zeros_(layer.bias)  # 偏置初始化为0

        # 对线性层的logit计算部分进行初始化
        init.xavier_uniform_(self.linear_Dlogit.weight)  # Xavier初始化
        init.zeros_(self.linear_Dlogit.bias)  # 偏置初始化为0

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for linear_dense_batch, linear_sparse_batch,dnn_dense_batch, dnn_sparse_batch, labels_batch in train_loader:
        linear_dense_batch = linear_dense_batch.to(device)
        linear_sparse_batch = linear_sparse_batch.to(device)
        dnn_dense_batch = dnn_dense_batch.to(device)
        dnn_sparse_batch = dnn_sparse_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(linear_dense_batch, linear_sparse_batch,
                        dnn_dense_batch, dnn_sparse_batch)
        loss = criterion(outputs.squeeze(), labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for linear_dense_batch, linear_sparse_batch,dnn_dense_batch, dnn_sparse_batch, labels_batch in val_loader:
            linear_dense_batch = linear_dense_batch.to(device)
            linear_sparse_batch = linear_sparse_batch.to(device)
            dnn_dense_batch = dnn_dense_batch.to(device)
            dnn_sparse_batch = dnn_sparse_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(linear_dense_batch, linear_sparse_batch,
                            dnn_dense_batch, dnn_sparse_batch)
            val_loss = criterion(outputs.squeeze(), labels_batch)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

def plot_losses(train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    path = '搜广推\\fun-rec\\codes\\base_models\\data\\criteo_sample.txt'
    data = pd.read_csv(path, sep=',')

    columns = data.columns.values # 提取列名作为列表以方便查找
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    train_data = process_data(data,dense_features, sparse_features)
    labels = data['label'].values

    # 准备输入字典
    dense_dim = 13
    sparse_dim = 26
    embed_dim = 4
    batch_size = 16
    epochs = 60
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 分别构建线性特征和dnn的特征
    linear_feature = {
        'sparse':[{'name':feat,'vocab_size':data[feat].nunique(),'embed_dim':1} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }

    dnn_feature = {
        'sparse':[{'name':feat,'vocab_size':data[feat].nunique(),'embed_dim':embed_dim} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }
    model = xDeepFM(linear_feature,dnn_feature,dense_dim,sparse_dim,embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # 使用 AdamW 优化器，并添加学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)  # weight_decay是L2正则化项

    # 使用学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 准备数据
    linear_dense_input = torch.tensor(train_data[dense_features].values, dtype=torch.float32)
    linear_sparse_input = torch.tensor(train_data[sparse_features].values, dtype=torch.long)
    dnn_dense_input = torch.tensor(train_data[dense_features].values, dtype=torch.float32)
    dnn_sparse_input = torch.tensor(train_data[sparse_features].values, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # 创建数据集和 DataLoader
    dataset = TensorDataset(linear_dense_input,linear_sparse_input,
                            dnn_dense_input,dnn_sparse_input,labels_tensor)

    # 划分训练集和验证集
    train_size = int(0.6 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 记录损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)

        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 调用学习率调度器
        scheduler.step(val_loss)

        # 每10轮输出一次损失
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 可视化损失
    plot_losses(train_losses, val_losses, epochs)
    df = pd.read_csv('搜广推\code_bymyself\losses\wideNdeep_val_losses.csv')
    df['xDeepFM'] = val_losses

    # 保存回 CSV 文件
    df.to_csv('搜广推\code_bymyself\losses\wideNdeep_val_losses.csv', index=False)

if __name__ == "__main__":
    main()
