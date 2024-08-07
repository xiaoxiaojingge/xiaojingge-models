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


import torch
import numpy as np

from kmeans_pytorch import kmeans
from text2vec import SentenceModel

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
# 分类类别数
class_num = 3

labels, cluster_centers = kmeans(
    X=corpus_embeddings,
    num_clusters=class_num,
    distance="euclidean",
    device=torch.device("cuda:0"),
)

class_data = {i: [] for i in range(class_num)}

for text, cls in zip(corpus, labels):
    class_data[cls.item()].append(text)

print(class_data)
