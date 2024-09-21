# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-10 14:01
@Author  : lijing
@File    : main.py
@Description: 
---------------------------------------
"""
import sys
from os import path

# pip3 install python-docx
from docx import Document
import chnSegment
import plot_wordcloud
import co_occurrence

"""
pip3 install jieba
pip3 install wordcloud
pip3 install python-docx
pip3 install matplotlib
pip3 install numpy
pip3 install seaborn
pip3 install networkx
"""

if __name__ == "__main__":

    # 读取文件 默认读取demo.txt 如有附加参数d 则读取demo.docx
    d = path.dirname(__file__)
    file_path = "doc/demo.txt"  # 要读取的文件名称
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # print(f"The first argument is: {first_arg}")
        if first_arg == "d":
            file_path = "doc/demo.docx"

            # 获取文件扩展名
    file_extension = path.splitext(file_path)
    content = ""
    if file_extension[1].lower() == ".txt":
        # 读取 .txt 文件
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    elif file_extension[1].lower() == ".docx":
        # 使用 python-docx 读取 .docx
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
    else:
        print("未知文件类型")

        # 若是中文文本，则先进行分词操作
    text = chnSegment.word_segment(content)

    # 生成词云
    plot_wordcloud.generate_wordcloud(text)

    # 共现矩阵
    co_occurrence.generate_pic(text)
