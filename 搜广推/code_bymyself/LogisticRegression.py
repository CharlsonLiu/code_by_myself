#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :LogisticRegression.py
@说明        :
@时间        :2024/10/21 19:20:49
@作者        :Liu Ziyue
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self,learning_rate = 1e-3,iterations = 500):
        """
        初始化Logistic回归模型的参数。
        
        参数:
        learning_rate (float): 学习率，用于梯度下降。
        num_iterations (int): 梯度下降的迭代次数。
        """
        self.learning_rate = learning_rate
        self.num_iterations = iterations
        self.w = None
        self.b = None

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def fit(self, X ,y):
        self.w = np.random.randn(X.shape[1])
        self.b = 0
        for i in range(self.num_iterations):
            # 更新sigmoid的预测值
            y_hat = self.sigmoid(np.dot(X,self.w) + self.b)

            # 计算损失
            loss = (-1 / len(X)) * np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

            # 计算梯度
            d_w = (1 / len(X)) *np.dot(X.T,(y_hat-y))

            d_b = (1 / len(X))*np.sum(y_hat-y)

            # 更新参数
            self.w -= self.learning_rate * d_w
            self.b -= self.learning_rate * d_b
            # 可以选择打印每次迭代的损失
            if i % 10 == 0:
                print(f"Iteration {i}: Loss = {loss}")

    def predict(self,X):
        y_hat = self.sigmoid(np.dot(X,self.w) + self.b)
        y_hat[y_hat>=0.5] = 1
        y_hat[y_hat <0.5] = 0
        return y_hat
    
    def score(self,y_pred,y):
        accuracy = (y_pred==y).sum()/len(y)
        return accuracy

def main():
    # 导入数据
    iris = load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.random.seed(42)
    model = LogisticRegression(learning_rate=0.06,iterations=500)
    model.fit(X_train,y_train)
    # 结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train = model.score(y_train_pred, y_train)
    score_test = model.score(y_test_pred, y_test)

    print('训练集Accuracy: ', score_train)
    print('测试集Accuracy: ', score_test)
    
    # 可视化决策边界
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.savefig('a.png', dpi=720)
    plt.show()

if __name__=='__main__':
    main()






