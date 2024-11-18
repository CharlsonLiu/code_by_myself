#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :DCN.py
@说明        :
@时间        :2024/11/12 18:03:40
@作者        :Liu Ziyue
'''


from typing import List
import warnings

warnings.filterwarnings("ignore")
from IPython import embed
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.functional as F
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


class CrossLayer(nn.Module):
    def __init__(self,embed_dim,layer_nums = 3):
        super(CrossLayer, self).__init__()
        self.layer_nums = 3
        self.embed_dim = embed_dim
        self.W  = nn.ParameterList([nn.Parameter(torch.randn(self.embed_dim,1)) for _ in range(self.layer_nums)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(self.embed_dim,1)) for _ in range(self.layer_nums)])


    def forward(self,input):
        # input_shape[batch,features*embed_dim]->[batch,features*embed_dim]
        x_0 = input.unsqueeze(2)
        x_l = input
        for i in range(self.layer_nums):
            _x_l = x_l.unsqueeze(2).transpose(1,2)
            x_0l = torch.einsum('bik,bkj->bij',x_0,_x_l)
            w,b = self.W[i].unsqueeze(0), self.b[i].unsqueeze(0).squeeze(2)
            cross = torch.einsum('bij,bjk->bik',x_0l,w).squeeze(2)
            x_l = cross + b + x_l
        
        return x_l


class DeepFM(nn.Module):
    def __init__(self,DNNFeature, dense_dim, sparse_dim, embed_dim):
        super(DeepFM, self).__init__()

        # 交叉层部分
        self.cross = CrossLayer(dense_dim + sparse_dim * embed_dim,3)

        # DNN部分
        self.dnn_sparse_feature = DNNFeature['sparse']

        self.dnn_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.dnn_sparse_feature
        ])
        
        self.dnn_layers = nn.Sequential(
            nn.Linear(dense_dim + sparse_dim * embed_dim, 256),  # 第一层神经元数量设为256
            nn.ReLU(),
            nn.LayerNorm(256),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.5),  # 添加Dropout来减少过拟合

            nn.Linear(256, 128),  # 中间层，使用128个神经元
            nn.ReLU(),
            nn.LayerNorm(128),  # 使用LayerNorm
            nn.Dropout(0.3),  # 添加Dropout

            nn.Linear(128, 64),  # 减少神经元数量至64
            nn.ReLU(),
            nn.LayerNorm(64),  # 使用LayerNorm
            nn.Dropout(0.3),  # 添加Dropout

            nn.Linear(64, 32),  # 进一步减少神经元数量至32
            nn.ReLU(),
            nn.LayerNorm(32),  # 使用LayerNorm
            nn.Dropout(0.1),  # 添加Dropout
        )

        self.norm = nn.LayerNorm(dense_dim + sparse_dim * embed_dim + 32)
        self.out = nn.Linear(dense_dim + sparse_dim * embed_dim + 32 , 1)

        # 添加权重初始化
        self.init_weights()

    def forward(self,dnn_dense_data, dnn_sparse_data):
        # 构建所需要的数据
        sparse = torch.cat([emb(dnn_sparse_data[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=1)
        input = torch.cat([dnn_dense_data, sparse], dim=1)

        # 特征交叉组合层
        cross = self.cross(input)
        
        # DNN层
        dnn = self.dnn_layers(input)


        output_logit = self.out(self.norm(torch.cat([cross,dnn], dim = 1)))

        return output_logit

    def init_weights(self):
        # 对Embedding层进行初始化
        for emb in self.dnn_embeddings:
            init.xavier_uniform_(emb.weight)

        # 对线性层进行初始化
        for layer in self.dnn_layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier初始化
                init.zeros_(layer.bias)  # 偏置初始化为0


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for dnn_dense_batch, dnn_sparse_batch, labels_batch in train_loader:
        dnn_dense_batch = dnn_dense_batch.to(device)
        dnn_sparse_batch = dnn_sparse_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(dnn_dense_batch, dnn_sparse_batch)
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
        for dnn_dense_batch, dnn_sparse_batch, labels_batch in val_loader:
            dnn_dense_batch = dnn_dense_batch.to(device)
            dnn_sparse_batch = dnn_sparse_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(dnn_dense_batch, dnn_sparse_batch)
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
    # 构建特征

    dnn_feature = {
        'sparse':[{'name':feat,'vocab_size':data[feat].nunique(),'embed_dim':embed_dim} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }
    model = DeepFM(dnn_feature,dense_dim,sparse_dim,embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # 使用 AdamW 优化器，并添加学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)  # weight_decay是L2正则化项

    # 使用学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 准备数据
    dense_input = torch.tensor(train_data[dense_features].values, dtype=torch.float32)
    sparse_input = torch.tensor(train_data[sparse_features].values, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # 创建数据集和 DataLoader
    dataset = TensorDataset(dense_input,sparse_input,labels_tensor)

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
    df['DCN'] = val_losses

    # 保存回 CSV 文件
    df.to_csv('搜广推\code_bymyself\losses\wideNdeep_val_losses.csv', index=False)

if __name__ == "__main__":
    main()
