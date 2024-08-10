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
from sklearn.metrics import pairwise_distances

# pip install scikit-learn-extra
# K-means 不支持自定义距离度量，但你可以使用 KMedoids 作为替代，它支持任意距离度量。
from sklearn_extra.cluster import KMedoids

from sklearn.feature_extraction.text import CountVectorizer

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


def preprocess_text(content_lines, stopwords):
    """
    对文本进行预处理
    :param content_lines:
    :param sentences:
    :param stopwords:
    :return:
    """
    sentences = []
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
        finally:
            pass
    return sentences


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
            n_clusters=self.class_num, max_iter=1000, init="k-means++", tol=1e-6
        )

        # 创建 KMedoids 模型，指定簇的数量和距离度量
        """
        在 KMedoids 聚类中，sklearn-extra 提供了对多种距离度量方式的支持。具体支持的距离度量包括：
        欧氏距离 ('euclidean')：
            这是最常见的距离度量，计算两点之间的直线距离。
        曼哈顿距离 ('manhattan')：
            计算两点在各维度上绝对差值的和，也称为城市街区距离。
        切比雪夫距离 ('chebyshev')：
            计算两点在各维度上最大绝对差值。
        闵可夫斯基距离 ('minkowski')：
            这是一个包含欧氏距离和曼哈顿距离的广义度量，参数 p 用于定义距离的度量类型。
            当 p=1 时，等同于曼哈顿距离。
            当 p=2 时，等同于欧氏距离。
        余弦相似度 ('cosine')：
            计算两向量的余弦夹角，适用于向量空间模型。
        汉明距离 ('hamming')：
            计算两个相同长度的字符串或向量之间的不同位数。
        杰卡德距离 ('jaccard')：
            用于计算集合之间的相似性和差异性，通常用于离散数据。
        """
        # self.model = KMedoids(
        #     n_clusters=self.class_num, max_iter=1000, init="k-medoids++", metric="euclidean"
        # )

        s = self.model.fit(tmp)
        # 输出类别中心
        """
        在 scikit-learn 的 KMeans 聚类算法中，cluster_centers_ 属性的结果是一个数组，其中包含了每个簇的中心（质心）。
        这个结果的长度与以下因素有关：
            簇的数量：cluster_centers_ 的第一个维度的长度等于你在 KMeans 模型中指定的簇的数量 (n_clusters)。每一行对应一个簇的中心。因此，如果你指定了 n_clusters=K，则 cluster_centers_ 将有 K 行。
            特征的数量：cluster_centers_ 的第二个维度的长度等于数据集中每个样本的特征数量。即，如果你的数据集有 m 个特征，那么每个簇的中心将有 m 个特征值。
        """
        logger.success(f"聚类中心：{self.model.cluster_centers_}")
        # 获取SSE（inertia）
        """
        SSE（Sum of Squared Errors），也称为簇内平方和（Within-Cluster Sum of Squares, WCSS），是K-means聚类算法的一个重要评估指标。它度量了数据点到其各自簇中心的距离平方和。
        具体来说，SSE的大小有以下几个方面的意义：
            簇的紧密度：较低的SSE表示数据点更接近于簇中心，说明簇的紧密度较高。即簇内部的数据点相对较集中，分布较均匀。
            模型的拟合度：较低的SSE通常意味着K-means模型更好地拟合了数据。模型的分类效果较好，簇内的数据点间的差异较小。
            簇的数量选择：SSE可以用来选择合适的簇数量（K）。通常使用“肘部法则”（Elbow Method）来选择最佳的K值。具体方法是绘制不同K值下的SSE曲线，寻找SSE下降速度显著减缓的拐点，这个拐点对应的K值通常被认为是较优的簇数量。
        """
        logger.success(f"SSE: {self.model.inertia_}")

        # 计算每行文本到所有类别中心的距离
        distances = pairwise_distances(tmp, self.model.cluster_centers_)
        print(distances)

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
# 处理语料，语料的处理结果存放在sentences，这个处理可以放在模型上游处理
sentences = preprocess_text(corpus, None)

print(sentences)

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
# net.plot(weight)


p = net.predict(weight)

class_data = {i: [] for i in range(class_num)}

for text, cls in zip(corpus, p):
    class_data[cls.item()].append(text)

print(class_data)

class_keywords = []
for i in range(class_num):
    # texts = class_data[i]  # 获取第i类别下的文本列表
    handle_sentences = preprocess_text(class_data[i], None)
    print(handle_sentences)
    # 使用列表推导遍历和收集数据
    texts = [word for sentence in handle_sentences for word in sentence.split(" ")]
    # print(texts)
    # 将文本列表转换为词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # 获取词袋模型中的所有词语
    words = vectorizer.get_feature_names_out()

    # 遍历词频矩阵的每一列，找到最高词频的词语
    max_freq_words = []
    for j in range(X.shape[1]):
        word_freq = X[:, j].sum()
        if word_freq > 0:
            max_freq_words.append((word_freq, words[j]))

    # 根据词频排序，并选择最高词频的词语
    max_freq_words.sort(reverse=True)
    top_word_freq = max_freq_words[0][0]
    top_word = max_freq_words[0][1]
    # 将词频和词语记录
    class_keywords.append([top_word_freq, top_word])

# 输出每个类别下的最高词频词
for i, keyword in enumerate(class_keywords):
    print("Class {}: {}".format(i, keyword))
