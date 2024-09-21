# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 19:47
@Author  : lijing
@File    : sklearn_lsa_scipy.py
@Description:
---------------------------------------
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba


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
            continue
        finally:
            pass
    return sentences


documents = [
    "猫喜欢玩具",
    "狗喜欢走路",
    "猫狗都喜欢家具",
]

# 处理语料，语料的处理结果存放在sentences，这个处理可以放在模型上游处理
documents = preprocess_text(documents, None)
print(documents)
count_vectorizer = CountVectorizer(min_df=1)
X = count_vectorizer.fit_transform(documents)
feature_name = count_vectorizer.get_feature_names_out()
print(feature_name)
print(X)
print(X.toarray())

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(documents)
print(tfidf_vectorizer.get_feature_names_out())
print(X)
print(X.toarray())

# 直接采用了scikit-learn中的Newsgroups数据集。这是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。
from sklearn.datasets import fetch_20newsgroups

# http://qwone.com/~jason/20Newsgroups/
# 本地配置下载fetch_20newsgroups：https://blog.csdn.net/weixin_44278512/article/details/88702719
# 修改 D:\workspace_coding\environment\anaconda3\envs\env_model_python_3_8_19\Lib\site-packages\sklearn\datasets\_twenty_newsgroups.py
# 最终缓存文件位置：C:\Users\xiaojingge\scikit_learn_data\20news-bydate_py3.pkz
categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
remove = ("headers", "footers", "quotes")
newsgroups_train = fetch_20newsgroups(
    subset="train", categories=categories, remove=remove, download_if_missing=True
)
newsgroups_test = fetch_20newsgroups(
    subset="test", categories=categories, remove=remove, download_if_missing=True
)

print(newsgroups_train.data[:5])

docs = newsgroups_train.data
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(docs)

# 使用 CountVectorizer 创建矩阵
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(docs)
# vocab = np.array(count_vectorizer.get_feature_names_out())

# 转换为 TF-IDF 矩阵
# tfidf_transformer = TfidfTransformer()
# tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

# SVD奇异值分解
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

# 注意将s从一维向量转换成对角矩阵(diagonal matrix)
# a = np.allclose(U.dot(np.diag(s)).dot(Vh), vectors.toarray())
# b = np.allclose(U.T.dot(U), np.eye(U.shape[0]))
# c = np.allclose(Vh.dot(Vh.T), np.eye(Vh.shape[0]))
# print(a, b, c)

# 使用 scipy.sparse.linalg.svds
k = 2
U_svds, s_svds, Vh_svds = svds(X, k=k)
print("Results from svds:")
print("U:\n", U_svds)
print("s:\n", s_svds)
print("Vh:\n", Vh_svds)

# 使用 scipy.linalg.svd
U_svd, s_svd, Vh_svd = svd(X.toarray(), full_matrices=False)
print("\nResults from svd:")
print("U:\n", U_svd[:, :k])
print("s:\n", s_svd[:k])
print("Vh:\n", Vh_svd[:k, :])
# def show_topics(a):
#     top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-k - 1:-1]]
#     topic_words = ([top_words(t) for t in a])
#     return [' '.join(t) for t in topic_words]
#
# print(show_topics(Vh[:k]))


# 使用 sklearn.decomposition.TruncatedSVD
svd = TruncatedSVD(n_components=k)
X_reduced = svd.fit_transform(X)
print("\nResults from TruncatedSVD:")
print("Components:\n", svd.components_)
print("Explained variance ratio:\n", svd.explained_variance_ratio_)
print("U:\n", np.dot(X_reduced, svd.components_))
print("s:\n", np.sqrt(svd.explained_variance_ratio_ * np.sum(svd.singular_values_**2)))
print("s:\n", svd.singular_values_)
print("Vh:\n", svd.components_)

# Truncated SVD
from sklearn import decomposition

U, s, Vh = decomposition.randomized_svd(X, k)
print(f"U：{U}")
print(f"s：{s}")
print(f"Vh：{Vh}")
