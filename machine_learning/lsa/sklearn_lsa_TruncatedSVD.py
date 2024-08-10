# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-10 20:47
@Author  : lijing
@File    : sklearn_lsa_TruncatedSVD.py
@Description: 
---------------------------------------
'''
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from sklearn.decomposition import TruncatedSVD

# 直接采用了scikit-learn中的Newsgroups数据集。这是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。
from sklearn.datasets import fetch_20newsgroups

# http://qwone.com/~jason/20Newsgroups/
# 本地配置下载fetch_20newsgroups：https://blog.csdn.net/weixin_44278512/article/details/88702719
# 修改 D:\workspace_coding\environment\anaconda3\envs\env_model_python_3_8_19\Lib\site-packages\sklearn\datasets\_twenty_newsgroups.py
# 最终缓存文件位置：C:\Users\xiaojingge\scikit_learn_data\20news-bydate_py3.pkz
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove, download_if_missing=True)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove, download_if_missing=True)

print(newsgroups_train.data[:5])

docs = newsgroups_train.data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print("--------转成权重---------")
print(X.shape)
print(X)
print("--------获取特征（单词）---------")
words = vectorizer.get_feature_names_out()
print(words)
print(len(words), "个特征（单词）")

topics = 4
svd = TruncatedSVD(n_components=topics)  # 潜在语义分析，设置话题个数
X1 = svd.fit_transform(X)  # 训练并进行转化
print("--------lsa奇异值---------")
print(svd.singular_values_)
print(f"--------{len(newsgroups_train.data)}个文本，在{topics}个话题向量空间下的表示---------")
print(X1)

pick_docs = 2  # 每个话题挑出最具代表性的文档个数
topic_docid = [X1[:, t].argsort()[:-(pick_docs + 1):-1] for t in range(topics)]
# argsort,返回排序后的序号
print(f"--------每个话题挑出{pick_docs}个最具代表性的文档---------")
print(topic_docid)

# print("--------lsa.components_---------")
# 话题向量空间
# print(lsa.components_)
pick_keywords = 3  # 每个话题挑出的关键词个数
topic_keywdid = [svd.components_[t].argsort()[:-(pick_keywords + 1):-1] for t in range(topics)]
print("--------每个话题挑出3个关键词---------")
print(topic_keywdid)

print("--------打印LSA分析结果---------")
for t in range(topics):
    print("话题 {}".format(t))
    print("\t 关键词：{}".format(", ".join(words[topic_keywdid[t][j]] for j in range(pick_keywords))))
    for i in range(pick_docs):
        print("\t\t 文档{}".format(i))
        print("\t\t", docs[topic_docid[t][i]])

# 获取U、S和V矩阵
U = X1  # U矩阵
S = svd.singular_values_  # S矩阵
V = svd.components_  # V矩阵

# 打印结果
print("U Matrix:\n", U)
print("S Matrix:\n", S)
print("V Matrix:\n", V)
