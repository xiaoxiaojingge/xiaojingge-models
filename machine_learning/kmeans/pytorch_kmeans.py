# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-07 20:13
@Author  : lijing
@File    : pytorch_kmeans.py
@Description: kmeans_pytorch实现聚类
---------------------------------------
"""

# pip install fast-pytorch-kmeans kmeans_pytorch text2vec

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import numpy as np

from kmeans_pytorch import kmeans, kmeans_predict
from text2vec import SentenceModel

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import jieba

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# 日志相关
from config.logger import Logger

# 路径相关
from pathlib import Path

# 邮件相关
from utils.email_util import EmailServer

logger = Logger().get_logger

# 配置相关
from config.config import Config

config = Config().get_project_config

# 读取.ini文件，这里的文件使用相对路径拼接
config.read(str(Path(__file__).resolve().parent.parent.parent) + "\config\config.ini")

models_path = config.get("models", "models_path")

# text2vec-base-chinese 需要到 huggingface 上下载， https://huggingface.co/shibing624/text2vec-base-chinese/tree/main
embedder = SentenceModel(
    model_name_or_path=f"{models_path}/text2vec-base-chinese",
)

# 读取语料
with open(f"corpus/corpus_train.txt", mode="r", encoding="utf-8") as file:
    # 读取文件内容，按行分割
    corpus = file.readlines()
corpus_embeddings = embedder.encode(corpus)
# 使用GPU加速
corpus_embeddings = torch.from_numpy(corpus_embeddings).to("cuda")

# 数据标准化（如果需要）
# data_mean = torch.mean(corpus_embeddings, dim=0)
# data_std = torch.std(corpus_embeddings, dim=0)
# data_normalized = (corpus_embeddings - data_mean) / data_std

data_normalized = corpus_embeddings

# 定义参数
num_clusters = 3
max_iters = 1000  # 自定义最大迭代次数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化
current_iter = 0
converged = False
prev_centers = None
cluster_centers = None
cluster_ids = None

"""
labels, cluster_centers = kmeans(
    X=corpus_embeddings,
    num_clusters=num_clusters,
    distance="euclidean",
    device=torch.device("cuda:0"),
)

class_data = {i: [] for i in range(num_clusters)}
for text, cls in zip(corpus, labels):
    class_data[cls.item()].append(text)
print(class_data)

"""

while not converged and current_iter < max_iters:
    # 执行KMeans
    cluster_ids, cluster_centers = kmeans(
        X=data_normalized,
        num_clusters=num_clusters,
        distance="euclidean",
        tol=1e-4,
        device=device,
    )

    # 检查收敛条件
    if prev_centers is not None:
        center_change = torch.norm(cluster_centers - prev_centers, p="fro")
        if center_change < 1e-4:
            converged = True
    prev_centers = cluster_centers
    current_iter += 1

print("Cluster Centers:\n", cluster_centers)
print("Labels:\n", cluster_ids)

# 执行 KMeans 预测
new_data_labels = kmeans_predict(
    X=data_normalized,
    cluster_centers=cluster_centers,
    distance="euclidean",
    device=device,
)
print("Predicted Labels:\n", new_data_labels)

# 将聚类中心移动到设备
cluster_centers = cluster_centers.to(device)


def compute_sse1(data, cluster_centers, cluster_ids):
    sse = 0.0
    for i, center in enumerate(cluster_centers):
        cluster_points = data[cluster_ids == i]
        sse += torch.sum((cluster_points - center) ** 2)
    return sse.item()


sse1 = compute_sse1(data_normalized, cluster_centers, cluster_ids)
print(f"SSE: {sse1}")


def compute_sse2(data, cluster_centers):
    # 将聚类中心移动到设备
    cluster_centers = cluster_centers.to(device)
    # 计算每个数据点到其簇中心的距离
    distances = torch.cdist(data, cluster_centers)
    # 找到每个数据点的最近簇中心
    min_distances, _ = torch.min(distances, dim=1)
    # 计算 SSE（平方距离和）
    sse = torch.sum(min_distances**2).item()
    return sse


sse2 = compute_sse2(data_normalized, cluster_centers)
print(f"SSE: {sse2}")

class_data = {i: [] for i in range(num_clusters)}
for text, cls in zip(corpus, cluster_ids):
    class_data[cls.item()].append(text)
print(class_data)
