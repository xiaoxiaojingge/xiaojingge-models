# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 12:00
@Author  : lijing
@File    : sklearn_lda_demo1.py
@Description: 
---------------------------------------
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# 多个文档构成的列表
documents = ["今天 天气 很好 啊", "今天 天气 确实 很好"]
count_vectorizer = CountVectorizer()
# 构造词频矩阵
cv = count_vectorizer.fit_transform(documents)
# 获取特征词
feature_names = count_vectorizer.get_feature_names_out()
# 词频矩阵
matrix = cv.toarray()
df = pd.DataFrame(matrix, columns=feature_names)
print(df)

# 结果
# 注意，这每一行有 5 个数字，但后面的 4 个才是词频，第一个数字是文档的标号（从 0 开始）
"""
   今天  天气   很好   确实
0   1    1     1     0
1   1    1     1     1
"""