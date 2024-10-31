#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :DeepCrossing.py
@说明        :
@时间        :2024/10/29 19:22:30
@作者        :Liu Ziyue
'''

from typing import Dict, List
import torch
import torch.nn as nn 
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from tqdm import tqdm

"""
构建残差连接模块，后面直接调用就可以了
"""
class ResidualBlock(nn.Module):
    def __init__(self,units):
        super(ResidualBlock, self).__init__()
        self.dnn1 = nn.Linear(units,256)
        self.dnn2 = nn.Linear(256,units)

    def forward(self, x):
        residual = x
        x = torch.relu(self.dnn1(x))
        x = self.dnn2(x)
        # 在与残差拼接后过激活函数
        x = residual + x
        return torch.relu(x)
    

class DeepCrossing(nn.Module):
    def __init__(self, dnn_feature_columns):
        super(DeepCrossing, self).__init__()
        self.sparse_features = dnn_feature_columns['sparse']

        # 为每一个稀疏特征构建嵌入层
        self.embeddings = nn.ModuleList([
            Embedding(feat['vocabulary_size'], feat['embedding_dim']) for feat in self.sparse_features
        ])

        # DNN部分
        self.residual_blocks = nn.ModuleList([ResidualBlock(117) for _ in range(3)])
        self.dnn_output = nn.Linear(117,1)
    
    def forward(self,dense_inputs, sparse_inputs):
        # 处理稀疏特征，得到系数特征的embedding。
        #nn.li 取出对应稀疏特征的序号，然后去除所有该特征下的用户，得到embedding
        sparse_embeddings = [emb(sparse_inputs[:,i]) for i ,emb in enumerate(self.embeddings)]
        sparse_output = torch.cat(sparse_embeddings,dim=1)

        # 拼接特征
        dnn_inputs = torch.cat([dense_inputs, sparse_output], dim=1)

        for block in self.residual_blocks:
            dnn_inputs = block(dnn_inputs)

        output = self.dnn_output(dnn_inputs)
        return torch.sigmoid(output)

def data_process(data_df:pd.DataFrame,dense_feat:List,sparse_feat:List):
    """
    简单处理特征，包括填充缺失值，数值处理，类别编码
    param data_df: DataFrame格式的数据
    param dense_feat: 数值特征名称列表
    param sparse_feat: 类别特征名称列表
    """
    # 填充缺失值
    data_df[dense_feat] = data_df[dense_feat].fillna(0.0)
    for f in dense_feat:
        # 对稠密特征中的每个特征做对数变换
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_feat] = data_df[sparse_feat].fillna('-1')
    for f in sparse_feat:
        # 对稀疏特征中的每个特征做onehot编码
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])  

    return data_df[dense_feat + sparse_feat] 

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for dense_batch, sparse_batch, labels_batch in train_loader:
        dense_batch = dense_batch.to(device)
        sparse_batch = sparse_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(dense_batch, sparse_batch)
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
        for dense_batch, sparse_batch, labels_batch in val_loader:
            dense_batch = dense_batch.to(device)
            sparse_batch = sparse_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(dense_batch, sparse_batch)
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
    
    columns = data.columns.values
    # 拆分稠密和稀疏特征
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # 拆分训练数据与标签
    train_data = data_process(data, dense_feat=dense_features, sparse_feat=sparse_features)
    labels = data['label'].values

    # 构建特征标记
    # 将稀疏与稠密矩阵分别保存成字典形式，方便后面生成embedding矩阵
    # 这是要学习的点，这样查找起来很容易
    dnn_feature_columns = {
        'sparse': [{'name': feat, 'vocabulary_size': data[feat].nunique(), 'embedding_dim': 4} for feat in sparse_features],
        'dense': [{'name': feat, 'dimension': 1} for feat in dense_features]
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 32

    # 模型初始化
    model = DeepCrossing(dnn_feature_columns).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 准备数据
    dense_input_tensor = torch.tensor(train_data[dense_features].values, dtype=torch.float32)
    sparse_input_tensor = torch.tensor(train_data[sparse_features].values, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # 创建数据集和 DataLoader
    dataset = TensorDataset(dense_input_tensor, sparse_input_tensor, labels_tensor)

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