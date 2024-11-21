#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :DIN.py
@说明        :
@时间        :2024/11/20 16:01:31
@作者        :Liu Ziyue
'''


import itertools
import os
from turtle import forward
import warnings

from sklearn.model_selection import train_test_split
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

import matplotlib.pyplot as plt


# 构建数据集，将数据变为embedding
class MovieDataset(Dataset):
    def __init__(self, data, feature_columns, sparse_list, dense_list):
        super().__init__()
        """
        初始化自定义数据集，将稀疏特征、稠密特征和变长稀疏特征通过嵌入操作转换。
        """
        # 特征列表
        self.sparse_features = feature_columns['sparse']
        self.dense_features = feature_columns['dense']
        self.var_len_sparse_features = feature_columns['VarLenSparseFeat']

        # 构建稀疏特征的嵌入层
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.sparse_features
        ])
        
        # 构建行为特征列表的嵌入层
        self.behavior_feature_embedding = nn.ModuleList([
            nn.Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.sparse_features if feat['name'] == 'movie_id'
        ])

        # 构建变长稀疏特征的嵌入层
        self.behavior_seq_embeddings = nn.ModuleList([
            nn.Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.var_len_sparse_features
        ])

        # 数据列
        self.sparse_data = data[sparse_list].values
        self.dense_data = data[dense_list].values
        self.behavior_seq_data = [
            [int(num) for num in l.split(',')] for l in data['hist_movie_id']
        ]
        self.behavior_data = data['movie_id'].values
        self.labels = data['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本，返回嵌入后的稀疏特征、稠密特征、变长稀疏特征和标签。
        """
        # 获取单个样本数据
        sparse_inputs = self.sparse_data[idx]
        dense_inputs = self.dense_data[idx]
        behavior_inputs = self.behavior_data[idx]
        behavior_seq_inputs = self.behavior_seq_data[idx]
        label = self.labels[idx]

        # 稀疏特征嵌入
        sparse_embeds = torch.stack([
            self.sparse_embeddings[i](torch.tensor(sparse_inputs[i], dtype=torch.long)) for i in range(len(sparse_inputs))
        ])

        # 稠密特征嵌入
        dense_embeds = torch.tensor(dense_inputs, dtype=torch.float32)

        # 行为特征嵌入
        behavior_embed = self.behavior_feature_embedding[0](torch.tensor(behavior_inputs, dtype=torch.long))

        # 行为序列特征
        behavior_seq_embeds = torch.stack([
            self.behavior_seq_embeddings[i](torch.tensor(behavior_seq_inputs, dtype=torch.long)) for i in range(len(self.var_len_sparse_features))
        ]).squeeze(0)

        # 标签
        label = torch.tensor(label, dtype=torch.float32)

        return {
            'sparse_embeds': sparse_embeds,
            'dense_embeds': dense_embeds,
            'behavior_embeds': behavior_embed,
            'behavior_seq_embeds': behavior_seq_embeds,
            'label': label
        }
    

class LocalActivationUnit(nn.Module):
    def __init__(self):
        super(LocalActivationUnit, self).__init__()
        # 定义隐藏层
        self.dnn = nn.ModuleList([
            nn.Linear(32,128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128,64),
            nn.LayerNorm(64),
            nn.PReLU()
        ])
        # 输出层
        self.linear = nn.Linear(64, 1)

    def forward(self, behavior_data, behavior_seq_data):
        # query: [batch, 1, embed_dim], keys & values: [batch, seq_len, embed_dim]
        query, keys = behavior_data, behavior_seq_data
        
        # 使用广播机制（broadcasting）进行扩展
        query_expanded = query.unsqueeze(1)  # [batch, 1, embed_dim]
        query_expanded = query_expanded.expand(-1, keys.size(1), -1)  # [batch, seq_len, embed_dim]

        # 拼接 query, keys, query-keys, query*keys
        att_input = torch.cat([query_expanded, keys, query_expanded - keys, query_expanded * keys], dim=-1)

        # 输入到 DNN
        att_out = att_input
        for layer in self.dnn:
            att_out = layer(att_out)

        # 输出层
        att_out = self.linear(att_out).squeeze(-1)  # 输出结果 [batch, seq_len]

        return att_out


class AttentionPoolingLayer(nn.Module):
    def __init__(self, hidden_units=(32, 128, 64), dropout_rate=0.5):
        super(AttentionPoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit()

    def forward(self, behavior_data, behavior_seq_data):
        query, keys = behavior_data, behavior_seq_data
        
        # 获取行为序列embedding的mask矩阵，标记非零元素为True
        key_masks = (keys[:, :, 0] != 0)

        # 获取对应的注意力权重
        att_weight = self.local_att(query, keys)

        # 创建padding的张量，用来标记行为序列中无效的位置
        padding = torch.zeros_like(att_weight)  # B x len

        # outputs: 表示padding之后的注意力权重 B x 1 x len
        outputs = torch.where(key_masks, att_weight, padding).unsqueeze(1)

        # 计算注意力加权之后的向量
        outputs = torch.einsum('bij,bjk->bik', outputs, keys).squeeze(1)

        return outputs


class DIN(nn.Module):
    def __init__(self):
        super(DIN, self).__init__()
        
        # 初始化 AttentionPoolingLayer 层
        self.att_pooling = AttentionPoolingLayer()

        # 定义 DNN 层，逐层堆叠
        self.dnn_layers = nn.ModuleList([
            nn.Linear(49, 100),
            nn.PReLU(),
            nn.Dropout(0.5),  # Dropout 层，防止过拟合
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Dropout(0.5)  # Dropout 层
        ])
        
        # 输出层
        self.out = nn.Linear(50, 1)

    def forward(self, sparse_data, dense_data, behavior_embeds, behavior_seq_embeds):
        # 数据分为三部分
        batch, _, _ = sparse_data.shape
        sparse_data = sparse_data.view(batch, -1)  # 展平稀疏数据

        # 处理注意力部分
        query, keys, values = behavior_embeds, behavior_seq_embeds, behavior_seq_embeds
        dnn_seq_input = self.att_pooling(query, keys)

        # 拼接稠密数据、稀疏数据和注意力加权后的序列输入
        dnn_input = torch.cat([dense_data, sparse_data, dnn_seq_input], dim=-1)

        # 通过 DNN 层进行前向传播
        dnn_out = dnn_input
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)

        # 输出层
        dnn_logits = self.out(dnn_out)

        return dnn_logits
        

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # 提取各个输入字段，并将它们发送到设备上
        sparse_embeds = batch['sparse_embeds'].to(device)
        dense_embeds = batch['dense_embeds'].to(device)
        behavior_embeds = batch['behavior_embeds'].to(device)
        behavior_seq_embeds = batch['behavior_seq_embeds'].to(device)
        labels_batch = batch['label'].to(device)

        # 将模型也送到 device 上
        model.to(device)

        # 调用模型进行前向传播
        outputs = model(sparse_embeds, dense_embeds, behavior_embeds, behavior_seq_embeds)

        # 计算损失并反向传播
        loss = criterion(outputs.squeeze(), labels_batch)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    total_val_loss = 0
    
    with torch.inference_mode():
        for batch in val_loader:
            # 提取数据并将它们送到设备
            sparse_embeds = batch['sparse_embeds'].to(device)
            dense_embeds = batch['dense_embeds'].to(device)
            behavior_embeds = batch['behavior_embeds'].to(device)
            behavior_seq_embeds = batch['behavior_seq_embeds'].to(device)
            labels_batch = batch['label'].to(device)

            # 调用模型进行前向传播
            outputs = model(sparse_embeds, dense_embeds, behavior_embeds, behavior_seq_embeds)

            # 计算验证损失
            val_loss = criterion(outputs.squeeze(), labels_batch)
            total_val_loss += val_loss.item()

    # 计算并返回平均验证损失
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

def save2csv(file_path,val_losses):
    # 如果文件不存在，创建一个新的 DataFrame
    if not os.path.exists(file_path):
        df = pd.DataFrame({'AFM': val_losses})  # 创建新的DataFrame
    else:
        # 如果文件存在，读取并追加数据
        df = pd.read_csv(file_path)
        df['DIN'] = val_losses

    # 保存回 CSV 文件
    df.to_csv(file_path, index=False)

def main():
    path = '搜广推\\fun-rec\\codes\\base_models\\data\\movie_sample.txt'
    samples_data = pd.read_csv(path, sep='\t', header = None)
    samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id", "label"]

    sparse_list = ['user_id', 'gender', 'age', 'movie_id', 'movie_type_id']
    dense_list = ['hist_len']

    # 准备输入字典
    embed_dim = 8
    batch_size = 64
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 分别构建线性特征和dnn的特征
    feature_columns = {
        'sparse': [
            {'name': 'user_id', 'vocab_size': max(samples_data["user_id"]) + 1, 'embed_dim': embed_dim},
            {'name': 'gender', 'vocab_size': max(samples_data["gender"]) + 1, 'embed_dim': embed_dim},
            {'name': 'age', 'vocab_size': max(samples_data["age"]) + 1, 'embed_dim': embed_dim},
            {'name': 'movie_id', 'vocab_size': max(samples_data["movie_id"]) + 1, 'embed_dim': embed_dim},
            {'name': 'movie_type_id', 'vocab_size': max(samples_data["movie_type_id"]) + 1, 'embed_dim': embed_dim}
        ],
        'dense': [
            {'name': 'hist_len', 'dimension': 1}
        ],
        'VarLenSparseFeat':[
            {'name': 'hist_movie_id', 'vocab_size':max(samples_data["movie_id"]) + 1, 'embed_dim': embed_dim, 'max_len': 50}
        ]
    }

    # 划分数据集
    train_data, val_data = train_test_split(samples_data, test_size=0.2, random_state=42)
    # 创建数据集实例
    train_dataset = MovieDataset(train_data, feature_columns, sparse_list, dense_list)
    val_dataset = MovieDataset(val_data, feature_columns, sparse_list, dense_list)

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DIN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # 使用 AdamW 优化器，并添加学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay= 1e-2)  # weight_decay是L2正则化项


    # 记录损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)

        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)


        # 每10轮输出一次损失
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 可视化损失
    plot_losses(train_losses, val_losses, epochs)
    
    file_path = '搜广推\code_bymyself\losses\DIN_val_losses.csv'

    save2csv(file_path,val_losses)


if __name__ == "__main__":
    main()
