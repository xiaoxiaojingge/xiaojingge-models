# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-20 10:19
@Author  : lijing
@File    : common_util.py
@Description: 公共工具类
---------------------------------------
"""

import zipfile


def count_lines(filename):
    """
    获取文件的总行数
    :param filename: 文件名称
    :return:
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = sum(1 for line in file)
    return lines


def extract_txt_from_zip(zip_file):
    """
    从压缩包中读取txt中的内容汇总到一个字符串列表中
    :param zip_file: zip文件数据
    :return: 字符串列表
    """
    file_contents = []
    file_zip = zipfile.ZipFile(zip_file, "r")
    for file_name in file_zip.namelist():
        if file_name.endswith(".txt"):
            txt_content = file_zip.read(file_name).decode("utf-8")
            file_contents.append(txt_content)
    return file_contents
