import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 定义自定义数据集
class NCFDataset(Dataset):
    def __init__(self, data):
        """初始化数据集
        参数:
            data (pd.DataFrame): 包含用户ID、电影ID和标签的DataFrame。
        """
        super().__init__()
        # df['col'].values：将列中的元素转为numpy数组
        self.user_ids = data['user_id'].values  # 用户ID数组
        self.movie_ids = data['movie_id'].values  # 电影ID数组
        self.labels = data['label'].values  # 标签（评分）数组

    def __len__(self):
        """返回数据集的大小
        返回:
            int: 数据集中的样本数量
        """
        return len(self.labels)

    def __getitem__(self, index):
        """获取指定索引的数据项
        参数:
            index (int): 数据项的索引。
        返回:
            tuple: 包含用户ID、电影ID和标签的张量。
        """
        user_id = torch.tensor(self.user_ids[index], dtype=torch.long)  # 将用户ID转换为张量
        movie_id = torch.tensor(self.movie_ids[index], dtype=torch.long)  # 将电影ID转换为张量
        label = torch.tensor(self.labels[index], dtype=torch.float32)  # 将标签转换为张量

        return user_id, movie_id, label  # 返回用户ID、电影ID和标签


# 定义神经协同过滤模型
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        """初始化模型
        参数:
            num_users (int): 用户的数量。
            num_items (int): 电影的数量。
            embedding_dim (int): 嵌入维度，默认值为8。
        """
        super().__init__()

        # 定义GMF部分的嵌入层
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)  # 用户嵌入层
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)  # 电影嵌入层

        # 定义MLP部分的嵌入层
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)  # 用户嵌入层
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)  # 电影嵌入层

        # 定义MLP全连接层
        self.mlp_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),  # 输入为两个嵌入的拼接
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 16),  # 输出为16维
            nn.ReLU()  # 激活函数
        )
        self.output_layer = nn.Linear(16 + embedding_dim, 1)  # 输出层，结合GMF和MLP的输出

    def forward(self, user_id, item_id):
        """前向传播
        参数:
            user_id (torch.Tensor): 用户ID张量。
            item_id (torch.Tensor): 电影ID张量。
        返回:
            torch.Tensor: 模型输出的预测值。
        """
        # GMF部分
        gmf_user_embed = self.gmf_user_embedding(user_id)  # 获取用户嵌入
        gmf_item_embed = self.gmf_item_embedding(item_id)  # 获取电影嵌入
        gmf_output = gmf_user_embed * gmf_item_embed  # 计算GMF输出

        # MLP部分
        mlp_user_embed = self.mlp_user_embedding(user_id)  # 获取用户嵌入
        mlp_item_embed = self.mlp_item_embedding(item_id)  # 获取电影嵌入
        mlp_input = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)  # 拼接用户和电影嵌入
        mlp_output = self.mlp_fc_layers(mlp_input)  # 通过MLP层

        # 合并GMF和MLP的输出
        out = torch.cat([gmf_output, mlp_output], dim=-1)  # 拼接输出

        output = self.output_layer(out)  # 通过输出层

        return output  # 返回最终输出


def train(model, device, train_loader, val_loader, loss_fn, optimizer, epochs=100):
    """训练模型
    参数:
        model (nn.Module): 待训练的模型。
        device (str): 设备类型（'cuda'或'cpu'）。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        loss_fn (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epochs (int): 训练轮数，默认值为100。
    返回:
        tuple: 包含训练损失和验证损失的列表。
    """
    model.to(device)  # 将模型移动到指定设备
    model.train()  # 设置为训练模式

    train_losses = []  # 保存每个epoch的训练损失
    val_losses = []    # 保存每个epoch的验证损失

    for epoch in tqdm(range(epochs)):
        total_loss = 0  # 记录当前epoch的总损失
        for user_id, movie_id, label in train_loader:
            # 将数据移动到指定设备
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)
            label = label.to(device)

            output = model(user_id, movie_id)  # 模型预测
            # 通过view(-1)将输出和标签展平为一维数组，确保形状匹配
            loss = loss_fn(output.view(-1), label.view(-1))  # 计算损失

            optimizer.zero_grad()  # 清空梯度
            total_loss += loss.item()  # 累加损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        avg_train_loss = total_loss / len(train_loader)  # 计算平均训练损失
        train_losses.append(avg_train_loss)  # 记录训练损失

        # 验证模型
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)  # 记录验证损失

        # 输出训练和验证损失
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {val_loss}')

    return train_losses, val_losses  # 返回训练和验证损失列表


def validate(model, val_loader, loss_fn, device):
    """验证模型
    参数:
        model (nn.Module): 待验证的模型。
        val_loader (DataLoader): 验证数据加载器。
        loss_fn (nn.Module): 损失函数。
        device (str): 设备类型（'cuda'或'cpu'）。
    返回:
        float: 平均验证损失。
    """
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置为评估模式

    val_loss = 0  # 初始化验证损失
    with torch.inference_mode():  # 关闭梯度计算
        for user_id, movie_id, label in val_loader:
            # 将数据移动到指定设备
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)
            label = label.to(device)

            val_output = model(user_id, movie_id)  # 模型预测
            # 通过view(-1)将输出和标签展平为一维数组，确保形状匹配
            loss = loss_fn(val_output.view(-1), label.view(-1))  # 计算损失

            val_loss += loss.item()  # 累加损失

    avg_val_loss = val_loss / len(val_loader)  # 计算平均验证损失
    print(f'Validation Loss: {avg_val_loss}')  # 输出验证损失
    return avg_val_loss  # 返回验证损失


def main():
    rnames = ['user_id','movie_id','rating','timestamp']
    data = pd.read_csv('搜广推\\fun-rec\\codes\\base_models\\data\\ml-1m\\ratings.dat', sep='::', engine='python', names=rnames)

    # 先实例化 LabelEncoder
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # 对 user_id 和 movie_id 列进行编码
    data['user_id'] = user_encoder.fit_transform(data['user_id'])
    data['movie_id'] = movie_encoder.fit_transform(data['movie_id'])

    data['label'] = data['rating']
    Mydataset = NCFDataset(data[['user_id', 'movie_id', 'label']])

    batch_size = 512
    epochs = 50
    train_size = int(0.8 * len(Mydataset))
    val_size = len(Mydataset) - train_size
    train_dataset, val_dataset = random_split(Mydataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    num_users = data['user_id'].nunique()
    num_items = data['movie_id'].nunique()
    model = NCFModel(num_users, num_items, embedding_dim=8)

    loss_fn = nn.MSELoss()
    op = torch.optim.Adam(params=model.parameters(),lr = 1e-3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_losses, val_losses = train(model, device, train_loader, val_loader, loss_fn, op, epochs)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
