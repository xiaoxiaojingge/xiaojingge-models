# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-13 19:00
@Author  : lijing
@File    : process.py
@Description: 处理数据
---------------------------------------
'''
import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('dataset/train.csv')

# 调整列的顺序，或者增删列
# df = df[['content', 'label']]

# 保存到新的CSV文件
# df.to_csv('output.csv', index=False)

# 去除content不是str类型的数据
# df = df[df['content'].apply(lambda x: isinstance(x, str))]
# df.to_csv('dataset/output.csv', index=False)

# 打乱数据并抽取20000条
sample_df = df.sample(n=200, random_state=42)  # random_state用于确保结果可重现

# # 保存测试集到新的CSV文件
# sample_df.to_csv('dataset/valid.csv', index=False)

# 取出content字段，然后结果为id,content，其中id自增，最终csv文件字段为 id,content
df = sample_df.reset_index().rename(columns={'index': 'id'})
df = df[['id', 'content']]
df.to_csv('dataset/test.csv', index=False)
