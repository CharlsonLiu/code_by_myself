#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :CF.py
@说明        :
@时间        :2024/10/23 16:30:03
@作者        :Liu Ziyue
'''

"""
User-based算法存在两个重大问题：

- 数据稀疏性
    * 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品
不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。
    * 这导致UserCF不适用于那些正反馈获取较困难的应用场景(如酒店预订， 大件物品购买等低频应用)。
- 算法扩展性
    * 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出TopN 相似用户
        该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加。
    * 故不适合用户数据量大的情况使用。
"""
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# 定义Jaccard相关系数
# 无法反映具体用户的评分喜好，只是用来判断用户是否会对某物品打分

def JaccardCoefficient(set_u:frozenset,set_v:frozenset)->float:
    # 计算两个用户交互物品集合的交集大小
    size_inner = len(set_u.intersection(set_v))
    
    # 计算两个用户交互物品并集大小
    size_union = len(set_u.union(set_v))

    return size_inner / size_union if size_union != 0 else 0

def cosine_smi(user_u:List,user_v:List):
    # 余弦相似度用来衡量两个向量的夹角，值域在[-1,1]
    # 夹角越小两个向量越相似
    # 计算公式为 cos_sim = u*v/(mod(u)*mod(v))
    # 返回的是一个对称矩阵
    return cosine_similarity([user_u,user_v])

def pearson_smi(user_u:List,user_v:List):
    # 在余弦相似度的基础上使用平均分进行修正，
    # 减少偏置项的影响
    # 也就是先对数据点进行中心化后再操作
    return pearsonr(user_u,user_v)

def loadData():
    users = {'Alice': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
             'user1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
             'user2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
             'user3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
             'user4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
             }
    return users

user_data = loadData()
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)),# 生成一个单位矩阵
    index=user_data.keys(),#为行列分配用户标签
    columns=user_data.keys(),
)

for u1 ,item1 in user_data.items():
    for u2,item2 in user_data.items():
        if u1 == u2:
            continue
        vec1,vec2=[],[]
        for item,rating1 in item1.items():
            rating2 = item2.get(item,-1)
            if rating2==-1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)

        # 因为是一个2x2的矩阵，不关心与自身的关系
        similarity_matrix[u1][u2] = np.corrcoef(vec1,vec2)[0][1]

print(similarity_matrix)

target_user = 'Alice'
num = 2
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.to_list()
print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')


weighted_scores = 0.
corr_values_sum = 0.

target_item = 'E'

target_mean = np.mean(list(user_data[target_user].values()))

for user in sim_users:
    corr_val = similarity_matrix[target_user][user]
    corr_values_sum += corr_val

    user_mean = np.mean(list(user_data[user].values()))
    weighted_scores += corr_val * (user_data[user][target_item] - user_mean)

target_item_pred = target_mean + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')