#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :AFM.py
@说明        :
@时间        :2024/11/13 10:49:26
@作者        :Liu Ziyue
'''


import itertools
from typing import List
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


class AFM_Layer(nn.Module):
    def __init__(self, att_dims=8,embed_dim = 4):
        super(AFM_Layer, self).__init__()
        self.att_dims = att_dims
        # 需要先初始化注意力权重矩阵，然后在训练过程中
        # 不断更新。真实的权重是经过softmax激活之后得到的
        self.att_W = nn.Parameter(torch.randn(embed_dim, att_dims) * 0.01)
        self.att_b = nn.Parameter(torch.zeros(att_dims))
        self.project_h = nn.Parameter(torch.randn(att_dims, 1) * 0.01)
        self.project_p = nn.Parameter(torch.randn(embed_dim, 1) * 0.01)

    def forward(self, att_input):
        B, n, k = att_input.shape  # B: batch size, n: feature count, k: embedding dim
        rows = []
        cols = []

        # 将att_input中的所有向量进行两两组合
        # 这个于前面的FM不同，这里没有对公式进行化简，因此，
        # 只能用组合的方式进行筛选，从n个数中选取所有的2元组
        # r，c为二元组的索引
        for r, c in itertools.combinations(range(n), 2):  # r / c 是每对特征的索引
            rows.append(att_input[:, r, :])  # 获取第r个特征的嵌入，形状是 B x k
            cols.append(att_input[:, c, :])  # 获取第c个特征的嵌入，形状是 B x k

        # 将每一对特征的嵌入进行拼接
        p = torch.cat(rows, dim=1)  # B x (n(n-1)/2) x k
        q = torch.cat(cols, dim=1)  # B x (n(n-1)/2) x k

        # 计算两两向量之间对应元素的乘积
        element_wise_product = p * q  # B x (n(n-1)/2) x k

        # 计算attention值，根据公式进行计算，获取真正的att权重
        element_wise_product=element_wise_product.view(B,-1,k)
        att_temp = F.relu(torch.matmul(element_wise_product, self.att_W) + self.att_b)  # B x (n(n-1)/2) x att_dims
        att_temp = torch.matmul(att_temp, self.project_h)  # B x (n(n-1)/2) x 1
        att_temp = F.softmax(att_temp, dim=2)  # B x (n(n-1)/2) x 1

        # 对每一对特征的注意力值与乘积进行加权求和
        att_out = torch.sum(att_temp * element_wise_product, dim=1)  # B x k
        # 得到att——logit
        att_logits = torch.matmul(att_out, self.project_p)  # B x 1

        return att_logits


class AFM(nn.Module):
    def __init__(self, Linearfeature, DNNFeature, dense_dim,
                    sparse_dim, embed_dim):
        super(AFM, self).__init__()

        # 线性层部分，这里和wide and deep是一样的
        # 稠密和稀疏特征分开处理，得到各自的logit，
        # 相加就是线性部分的logit
        self.linear_sparse_feat = Linearfeature['sparse']

        self.linear_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.linear_sparse_feat
        ])

        # 构建线性部分的稠密logit计算模块
        self.linear_Dlogit = nn.Linear(dense_dim, 1)

        # 双线性池化部分,唯一不同的部分
        self.afm = AFM_Layer()

        # DNN部分
        self.dnn_sparse_feature = DNNFeature['sparse']

        self.dnn_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.dnn_sparse_feature
        ])
        
        self.dnn_layers = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout的比例增加
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 增加Dropout
            nn.LayerNorm(16),
            nn.Linear(16, 1),
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
        
        # 注意力部分，单独处理sparse部分，
        # 因此可以借用DNN部分的embedding
        att_input = torch.stack([emb(dnn_sparse_data[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=1)
        att_logits = self.afm(att_input)

        output_logit = linear_logit + att_logits

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
    epochs = 30
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
    model = AFM(linear_feature,dnn_feature,dense_dim,sparse_dim,embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # 使用 AdamW 优化器，并添加学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)  # weight_decay是L2正则化项

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

if __name__ == "__main__":
    main()
