# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 12:00
@Author  : lijing
@File    : sklearn_lda_demo4.py
@Description: 
---------------------------------------
"""
import pandas as pd
import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 下面的 url 是 csv 文件的远程链接，如果你缺失这个文件，则需要用浏览器打开这个链接
# 下载它，然后放到代码运行命令，且文件名应与下面的 csv_path 一致
url = "https://raw.githubusercontents.com/Micro-sheep/Share/main/zhihu/answers.csv"
# 本地 csv 文档路径
csv_path = "answers.csv"
# 待分词的 csv 文件中的列
document_column_name = "回答内容"
pattern = '[\\s\\d,.<>/?:;\'"[\\]{}()\\|~!\t"@#$%^&*\\-_=+，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'

# if not os.path.exists(csv_path):
#     print(
#         f"请用浏览器打开 {url} 并下载该文件(如果没有自动下载，则可以在浏览器中按键盘快捷键 ctrl s 来启动下载)"
#     )
# df = pd.read_csv(csv_path, encoding="utf-8-sig")
# df = df.drop_duplicates()
# df = df.rename(columns={document_column_name: "text"})
# df["cut"] = df["text"].apply(lambda x: str(x))
# df["cut"] = df["cut"].apply(lambda x: re.sub(pattern, " ", x))
# df["cut"] = df["cut"].apply(lambda x: " ".join(jieba.lcut(x)))
# print(df["cut"])


if not os.path.exists(csv_path):
    print(
        f"请用浏览器打开 {url} 并下载该文件(如果没有自动下载，则可以在浏览器中按键盘快捷键 ctrl s 来启动下载)"
    )
df = (
    pd.read_csv(csv_path, encoding="utf-8-sig")
    .drop_duplicates()
    .rename(columns={document_column_name: "text"})
)

# 去重、去缺失、分词
df["cut"] = (
    df["text"]
    .apply(lambda x: str(x))
    .apply(lambda x: re.sub(pattern, " ", x))
    .apply(lambda x: " ".join(jieba.lcut(x)))
)
# print(df["cut"])


# 构造 TF-IDF
tf_idf_vectorizer = TfidfVectorizer()
tf_idf = tf_idf_vectorizer.fit_transform(df["cut"])
# 特征词列表
feature_names = tf_idf_vectorizer.get_feature_names_out()
# 特征词 TF-IDF 矩阵
matrix = tf_idf.toarray()
feature_names_df = pd.DataFrame(matrix, columns=feature_names)
# print(feature_names_df)


from sklearn.decomposition import LatentDirichletAllocation

# 指定 lda 主题数
n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)
print(lda)

# 核心，给 LDA 喂生成的 TF-IDF 矩阵
lda.fit(tf_idf)


def top_words_data_frame(
    model: LatentDirichletAllocation,
    tf_idf_vectorizer: TfidfVectorizer,
    n_top_words: int,
) -> pd.DataFrame:
    """
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    """
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names()
    for topic in model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        rows.append(top_words)
    columns = [f"topic {i+1}" for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df


def predict_to_data_frame(
    model: LatentDirichletAllocation, X: np.ndarray
) -> pd.DataFrame:
    """
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    """
    # 求出给定文档的主题概率分布矩阵
    matrix = model.transform(X)
    columns = [f"P(topic {i+1})" for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df
