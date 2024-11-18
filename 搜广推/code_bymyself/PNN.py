#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :PNN.py
@说明        :
@时间        :2024/10/30 18:28:58
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


def data_process(data_df:pd.DataFrame,dense_feat:List,sparse_feat:List):
    """
    简单处理特征，包括填充缺失值，数值处理，类别编码
    param data_df: DataFrame格式的数据
    param dense_feat: 数值特征名称列表
    param sparse_feat: 类别特征名称列表
    """
    data_df[dense_feat] = data_df[dense_feat].fillna(0.0)
    for f in dense_feat:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_feat] = data_df[sparse_feat].fillna('-1')
    for f in sparse_feat:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_feat + sparse_feat]

# 由于IPNN的复杂度远小于OPNN，因此在工业界中只可能使用IPNN
# 所以本代码只实现IPNN的部分
class ProductLayer(nn.Module):
    def __init__(self,in_features:int,embed_dim:int,out_features:int):
        super(ProductLayer, self).__init__()
        
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.linear_w = nn.Linear(in_features * embed_dim, out_features)

        self.inner_w = nn.Parameter(torch.randn(out_features, in_features))
        

    def forward(self, inputs):
        # inputs shape:[batch,in_features,embed_dim]
        # 获取线性层的输出
        lz = self.linear_w(inputs)
        # 获取非线性层的输出
        lp_list = []
        for i in range(self.out_features):
            # 提取第i个unit的权重值
            weight_row = self.inner_w[i]
            # 使用爱因斯坦求和实现逐元素相乘求和
            # # 'bfe, f -> be' 表示：batch (b), features (f), embed_dim (e) 上加权求和
            delta = torch.einsum('bfe,f->be',inputs.view(-1,self.in_features,self.embed_dim),weight_row) # shape: [batch, embed_dim]

            lp_list.append(torch.norm(delta,p = 2, dim=1))

        # 将所有输出结果堆叠为 [batch, out_features] 的输出
        lp = torch.stack(lp_list,dim=1)
        output = torch.cat([lz,lp], dim=1)
        return output


class PNN(nn.Module):
    def __init__(self, dnn_feature_columns,in_features,embed_dim,out_features):
        super(PNN, self).__init__()
        self.sparse_feat = dnn_feature_columns['sparse']
        # 为每一个稀疏特征构建嵌入层
        self.embeddings = nn.ModuleList([
            Embedding(feat['vocabulary_size'], feat['embedding_dim']) for feat in self.sparse_feat
        ])

        # PNN部分
        self.prodcutLayer=ProductLayer(in_features,embed_dim,out_features)

        self.linear_blocks = nn.Sequential(
            nn.Linear(out_features * 2,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        # 处理稀疏特征，得到系数特征的embedding list。
        # 取出对应稀疏特征的序号，然后去除所有该特征下的用户，得到embedding
        sparse_embeddings = [emb(inputs[:,i]) for i ,emb in enumerate(self.embeddings)]
        sparse_output = torch.cat(sparse_embeddings,dim=1)
        print(sparse_output.shape)

        prodcut_output = self.prodcutLayer(sparse_output)
        dnn_out = self.linear_blocks(prodcut_output)
        return dnn_out


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for sparse_batch, labels_batch in tqdm(train_loader):
        sparse_batch = sparse_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(sparse_batch)
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
        for sparse_batch, labels_batch in tqdm(val_loader):
            sparse_batch = sparse_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(sparse_batch)
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
    
    # Split dense and sparse features
    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # Process data
    train_data = data_process(data, dense_features, sparse_features)
    labels = data['label'].values

    # Prepare the input dictionary
    embed_dim = 4
    out_dim = 16
    batch_size = 16
    epochs = 60
    dnn_feature_columns = {
        'sparse': [{'name': feat, 'vocabulary_size': data[feat].nunique(), 'embedding_dim': embed_dim} for feat in sparse_features],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pnn = PNN(dnn_feature_columns,in_features=len(sparse_features),embed_dim=embed_dim,out_features=out_dim).to(device)
    loss = nn.BCELoss()
    op = torch.optim.Adam(params=pnn.parameters(),lr = 5e-5,weight_decay=1e-3)

    labels_tensor = torch.tensor(labels,dtype=torch.float32)
    sparse_input_tensor = torch.tensor(train_data[sparse_features].values,dtype=torch.long)
    dataset = TensorDataset(sparse_input_tensor,labels_tensor)
    
    # 划分训练集和验证集
    train_size = int(0.6 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 记录损失
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        avg_loss = train_epoch(pnn, train_loader, loss, op, device)
        train_losses.append(avg_loss)

        # 验证
        val_loss = validate_epoch(pnn, val_loader, loss, device)
        val_losses.append(val_loss)

        # 每10轮输出一次损失
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 可视化损失
    plot_losses(train_losses, val_losses, epochs)

    df = pd.read_csv('搜广推\code_bymyself\losses\wideNdeep_val_losses.csv')
    df['PNN'] = val_losses

    # 保存回 CSV 文件
    df.to_csv('搜广推\code_bymyself\losses\wideNdeep_val_losses.csv', index=False)

if __name__ =='__main__':
    main()