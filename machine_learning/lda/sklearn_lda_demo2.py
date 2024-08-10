# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 12:00
@Author  : lijing
@File    : sklearn_lda_demo2.py
@Description: 
---------------------------------------
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 多个文档构成的列表
documnets = ["今天 天气 很好 啊", "今天 天气 确实 很好"]
tf_idf_vectorizer = TfidfVectorizer()
# 构造词频矩阵
tf_idf = tf_idf_vectorizer.fit_transform(documnets)
# 获取特征词
feature_names = tf_idf_vectorizer.get_feature_names_out()
# 词频矩阵
matrix = tf_idf.toarray()
df = pd.DataFrame(matrix, columns=feature_names)
print(df)

# 结果
# 注意，这每一行有 5 个数字，但后面的 4 个才是词频，第一个数字是文档的标号（从 0 开始）
"""
         今天        0天气        很好        确实
0     0.577350     0.57735    0.577350    0.000000
1     0.448321     0.448321   0.448321    0.630099
"""
