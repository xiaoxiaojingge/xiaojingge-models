# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-12 16:42
@Author  : lijing
@File    : config.py
@Description: 配置相关，里面的参数信息，根据实际的情况调整
---------------------------------------
"""

import torch

"""
vocab_size：词汇表的大小。

embed_size：词嵌入的维度。

num_classes：分类的类别数量。

kernel_sizes：卷积核的大小列表。

num_channels：每个卷积核的数量（输出通道数）。

dropout：dropout 概率。

batch_size：每批次处理的样本数量。

lr：学习率。

num_epochs：训练的迭代次数。

num_workers：数据加载时的线程数量。
"""


class Config:
    def __init__(self):
        self.vocab_size = 30000
        self.embed_size = 300
        self.num_classes = 10
        self.kernel_sizes = [3, 4, 5]
        self.num_channels = 100
        self.dropout = 0.5
        self.batch_size = 64
        self.lr = 1e-3
        self.num_epochs = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 0  # 线程数，根据需要调整
