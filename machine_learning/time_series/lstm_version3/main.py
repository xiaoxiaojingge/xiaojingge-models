# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-17 8:12
@Author  : lijing
@File    : main.py
@Description: 
---------------------------------------
"""


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader


def predict_next(model, sample, epoch=20):
    temp1 = list(sample[:, 0])
    for i in range(epoch):
        sample = sample.reshape(1, x_Seq_len, 1)
        pred = model.predict(sample)
        value = pred.tolist()[0][0]
        temp1.append(value)
        sample = np.array(temp1[i + 1 : i + x_Seq_len + 1])
    return temp1


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_batch_data(X, y, batch_size=32, data_type=1):
    """基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）
    Returns:
        train_batch_data 或 test_batch_data
    """
    dataset = TimeSeriesDataset(X, y)

    if data_type == 1:  # 测试集
        test_batch_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_batch_data
    else:  # 训练集
        train_batch_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_batch_data


def create_new_dataset(dataset, seq_len=12):
    """基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）
    Returns:
        X, y
    """
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表,y标签为样本的下一个点，即预测点

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置,dataset.shape[0]就是有多少条

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i : i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label
    # 返回特征数据集和标签集
    return np.array(X), np.array(y)


def split_dataset(X, y, train_ratio=0.8):
    """基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_len = len(X)  # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio)  # 训练集的样本数量

    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集

    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    # 返回值
    return X_train, X_test, y_train, y_test


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out


def create_batch_data(X, y, batch_size=32, data_type=1):
    dataset = TimeSeriesDataset(X, y)
    shuffle = data_type == 2
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    x_Seq_len = 32

    dataset = pd.read_csv("./dataset/time_series_data.csv")
    dataset = dataset.iloc[:, [0, 1]]
    dataset.columns = ["date", "value"]
    dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d")
    dataset.index = dataset.date
    dataset.drop(columns="date", axis=1, inplace=True)

    plt.figure()
    plt.plot(dataset)
    plt.show()

    print(dataset.info())

    f, ax = plt.subplots()
    sns.boxplot(y="value", data=dataset, ax=ax)
    plt.show()

    s = dataset.describe()
    print(s)
    q1 = s.loc["25%"]
    q3 = s.loc["75%"]
    iqr = q3 - q1
    mi = q1 - 1.5 * iqr
    ma = q3 + 1.5 * iqr

    print("最大值：", ma, "最小值：", mi)
    dataset = dataset.drop(
        index=dataset[((dataset.value > ma) | (dataset.value < mi)).index]
    )

    scaler = MinMaxScaler()
    dataset["OT"] = scaler.fit_transform(dataset["OT"].values.reshape(-1, 1))

    dataset["OT"].plot()
    plt.show()

    dataset_new = dataset
    X, y = create_new_dataset(dataset_new.values, seq_len=x_Seq_len)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    test_batch_dataset = create_batch_data(X_test, y_test, batch_size=24, data_type=1)
    train_batch_dataset = create_batch_data(
        X_train, y_train, batch_size=24, data_type=2
    )

    input_dim = 1
    hidden_dim = 8
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, output_dim)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_batch_dataset:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X_batch, y_batch in test_batch_dataset:
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {test_loss / len(test_batch_dataset):.4f}"
        )

    plt.figure()
    plt.plot([loss for X_batch, y_batch in train_batch_dataset], label="train loss")
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

    model.eval()
    with torch.no_grad():
        test_pred = []
        true_vals = []
        for X_batch, y_batch in test_batch_dataset:
            outputs = model(X_batch)
            test_pred.extend(outputs.numpy())
            true_vals.extend(y_batch.numpy())

    plt.figure()
    plt.plot(true_vals, label="True")
    plt.plot(test_pred, label="Pred")
    plt.legend(["True", "Pred"])
    plt.show()

    score = r2_score(true_vals, test_pred)
    print("r^2 的值： ", score)
    # 绘制test中前100个点的真值与预测值
    y_true = y_test  # 真实值
    y_pred = test_pred  # 预测值

    fig, axes = plt.subplots(2, 1)
    ax0 = axes[0].plot(y_true, marker="o", color="red", label="true")
    ax1 = axes[1].plot(y_pred, marker="*", color="blue", label="pred")
    plt.show()
    """
    模型测试
    """
    # 选择test中的最后一个样本
    sample = X_test[-1]
    sample = sample.reshape(1, sample.shape[0], 1)
    # 模型预测
    sample_pred = model.predict(sample)  # predict()预测标签值
    ture_data = X_test[-1]  # 真实test的最后20个数据点
    # 预测后48个点
    preds = predict_next(model, ture_data, 48)
    # 绘图
    plt.figure()
    plt.plot(preds, color="yellow", label="Prediction")
    plt.plot(ture_data, color="blue", label="Truth")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()
relative_error = 0
"""模型精确度计算"""
for i in range(len(y_pred)):
    relative_error += (abs(y_pred[i] - y_true[i]) / y_true[i]) ** 2
acc = 1 - np.sqrt(relative_error / len(y_pred))
print(f"模型的测试准确率为：", acc)
