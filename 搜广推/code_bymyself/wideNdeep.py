#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :wideNdeep.py
@说明        :
@时间        :2024/11/05 19:19:53
@作者        :Liu Ziyue
'''

from typing import List
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,TensorDataset

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

class WideAndDeep(nn.Module):
    def __init__(self,Linearfeature,DNNFeature,dense_dim,sparse_dim,out_dim):
        super(WideAndDeep, self).__init__()
        # 线性层部分
        self.linear_sparse_feat = Linearfeature['sparse']

        self.linear_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'], feat['embed_dim']) for feat in self.linear_sparse_feat
        ])

        # 构建线性部分的稠密logit计算模块
        self.linear_Dlogit = nn.Linear(dense_dim,1)

        # DNN部分
        self.dnn_sparse_feature = DNNFeature['sparse']

        self.dnn_embeddings = nn.ModuleList([
            Embedding(feat['vocab_size'],feat['embed_dim']) for feat in self.dnn_sparse_feature 
        ])
        self.dnn_layers = nn.Sequential(
            nn.Linear(x,y),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(y,z),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(z,d),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            nn.Linear(d,1)
        )

    def forward(self,linear_dense_data,linear_sparse_data,dnn_dense_data,dnn_sparse_data):
        # 线性层
        # 构建embedding
        linear_sparse = torch.cat([emb(linear_sparse_data[:,i]) for i ,emb in enumerate(self.linear_embeddings)],dim = 1)
        linear_sparse_logits = linear_sparse.sum(dim = 1)
        linear_dense_logits = self.linear_Dlogit(torch.cat(linear_dense_data,dim=1))
        linear_logit = torch.sum(linear_dense_logits,linear_sparse_logits)
        
        # DNN层
        dnn_sparse = torch.cat([emb(dnn_sparse_data[:,i]) for i, emb in enumerate(self.dnn_embeddings)],dim=1)
        dnn_input = torch.cat([dnn_dense_data,dnn_sparse],dim = 1)
        dnn_logit = self.dnn_layers(dnn_input)

        output_logit = linear_logit + dnn_logit

        return nn.Sigmoid(output_logit)



def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for linear_dense_batch, linear_sparse_batch,\
        dnn_dense_batch, dnn_sparse_batch, labels_batch in train_loader:
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
        for linear_dense_batch, linear_sparse_batch,\
            dnn_dense_batch, dnn_sparse_batch, labels_batch in val_loader:
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
    embed_dim = 4
    out_dim = 16
    batch_size = 32
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 分别构建线性特征和dnn的特征
    linear_feature = {
        'sparse':[{'name':feat,'vocab_size':data[feat].nunique,'embed_dim':1} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }

    dnn_feature = {
        'sparse':[{'name':feat,'vocab_size':data[feat].nunique,'embed_dim':embed_dim} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }
    model = WideAndDeep(linear_feature,dnn_feature,dense_dim,sparse_dim,out_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
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
    train_size = int(0.8 * len(dataset))
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

        # 每10轮输出一次损失
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 可视化损失
    plot_losses(train_losses, val_losses, epochs)

if __name__ == "__main__":
    main()
