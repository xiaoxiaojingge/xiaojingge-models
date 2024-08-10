# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 12:24
@Author  : lijing
@File    : sklearn_lda_demo3.py
@Description: 
---------------------------------------
"""
import pandas as pd
import os

# 下面的 url 是 csv 文件的远程链接，如果你缺失这个文件，则需要用浏览器打开这个链接
# 下载它，然后放到代码运行命令，且文件名应与下面的 csv_path 一致
url = "https://raw.githubusercontents.com/Micro-sheep/Share/main/zhihu/answers.csv"
csv_path = "answers.csv"
if not os.path.exists(csv_path):
    print(
        f"请用浏览器打开 {url} 并下载该文件(如果没有自动下载，则可以在浏览器中按键盘快捷键 ctrl s 来启动下载)"
    )

df = pd.read_csv(csv_path)
print(list(df.columns))
