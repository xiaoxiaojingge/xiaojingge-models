# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-07 21:32
@Author  : lijing
@File    : sklearn_kmeans.py
@Description: TF-IDF sklearn 聚类
---------------------------------------
"""

import re
import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing
import os


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


def preprocess_text(content_lines, sentences, stopwords):
    """
    对文本进行预处理
    :param content_lines:
    :param sentences:
    :param stopwords:
    :return:
    """
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            # 去数字
            segs = [v for v in segs if not str(v).isdigit()]
            # 去左右空格
            segs = list(filter(lambda x: x.strip(), segs))
            # 长度为1的字符
            segs = list(filter(lambda x: len(x) > 1, segs))
            # 去掉停用词
            if stopwords:
                segs = list(filter(lambda x: x not in stopwords, segs))
            sentences.append(" ".join(segs))
        except Exception as e:
            logger.error(e)
            logger.info(line)
            continue


class JieKmeans:
    def __init__(self, class_num=4, n_components=10, func_type="PCA"):
        self.model = None
        self.PCA = PCA(n_components=n_components)
        if func_type == "PCA":
            self.func_plot = PCA(n_components=n_components)
        elif func_type == "TSNE":
            from sklearn.manifold import TSNE

            self.func_plot = TSNE(n_components=n_components, perplexity=8)
        self.class_num = class_num

    def plot_cluster(self, result, newData):
        plt.figure(2)
        Lab = [[] for i in range(self.class_num)]
        index = 0
        for labi in result:
            Lab[labi].append(index)
            index += 1
        color = [
            "oy",
            "ob",
            "og",
            "cs",
            "ms",
            "bs",
            "ks",
            "ys",
            "yv",
            "mv",
            "bv",
            "kv",
            "gv",
            "y^",
            "m^",
            "b^",
            "k^",
            "g^",
        ] * 3

        for i in range(self.class_num):
            x1 = []
            y1 = []
            for ind1 in newData[Lab[i]]:
                try:
                    y1.append(ind1[1])
                    x1.append(ind1[0])
                except:
                    pass
            plt.plot(x1, y1, color[i])

        # 绘制初始中心点
        x1 = []
        y1 = []
        for ind1 in self.model.cluster_centers_:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, "rv")  # 绘制中心
        plt.show()

    def train(self, data):
        tmp = self.PCA.fit_transform(data)
        self.model = KMeans(
            n_clusters=self.class_num, max_iter=10000, init="k-means++", tol=1e-6
        )
        s = self.model.fit(tmp)
        logger.success("聚类算法训练完成！\n", s)

    def predict(self, data):
        t_data = self.PCA.fit_transform(data)
        result = list(self.model.predict(t_data))
        return result

    def plot(self, weight):
        t_data = self.PCA.fit_transform(weight)
        result = list(self.model.predict(t_data))
        plot_pos = self.func_plot.fit_transform(weight)
        self.plot_cluster(result, plot_pos)


# 读取语料
with open(f"corpus/corpus_train.txt", mode="r", encoding="utf-8") as file:
    # 读取文件内容，按行分割
    corpus = file.readlines()

# 添加自定义词
jieba.add_word("花呗")
sentences = []
# 处理语料，语料的处理结果存放在sentences，这个处理可以放在模型上游处理
preprocess_text(corpus, sentences, None)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names_out()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 查看特征大小
# print("Features length: " + str(len(word)))

class_num = 3
net = JieKmeans(class_num=class_num, n_components=5, func_type="PCA")
net.train(weight)

# 绘制聚类结果
net.plot(weight)


p = net.predict(weight)

class_data = {i: [] for i in range(class_num)}

for text, cls in zip(corpus, p):
    class_data[cls.item()].append(text)

print(class_data)
