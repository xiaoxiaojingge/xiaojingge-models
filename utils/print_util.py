#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

"""
---------------------------------------
@Time    : 2023-09-18
@Author  : lijing
@File    : print_util.py
@Description: 打印工具类
---------------------------------------
"""

# 如果你想要输出颜色化的文本，可以使用Python内置的colorama模块
import colorama

colorama.init()


# 打印默认信息
def print_default(message):
    print(f"\033[96m{message}\033[0m")


# 打印成功信息
def print_success(message):
    print(f"\033[92m{message}\033[0m")


# 打印警告信息
def print_warn(message):
    print(f"\033[93m{message}\033[0m")


# 打印错误信息
def print_error(message):
    print(f"\033[91m{message}\033[0m")


# 打印完成信息
def print_complete(message):
    print(f"\033[94m{message}\033[0m")
