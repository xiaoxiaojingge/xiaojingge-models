# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-21 14:52
@Author  : lijing
@File    : config.py
@Description: 配置类
---------------------------------------
"""

# 配置
import configparser
from functools import wraps

# 路径相关
from pathlib import Path


# 单例类的装饰器
def singleton_class_decorator(cls):
    """
    装饰器，单例类的装饰器
    """
    # 在装饰器里定义一个字典，用来存放类的实例。
    _instance = {}

    # 装饰器，被装饰的类
    @wraps(cls)
    def wrapper_class(*args, **kwargs):
        # 判断，类实例不在类实例的字典里，就重新创建类实例
        if cls not in _instance:
            # 将新创建的类实例，存入到实例字典中
            _instance[cls] = cls(*args, **kwargs)
        # 如果实例字典中，存在类实例，直接取出返回类实例
        return _instance[cls]

    # 返回，装饰器中，被装饰的类函数
    return wrapper_class


@singleton_class_decorator
class Config:
    def __init__(self):
        """
        构造方法
        """
        config = configparser.ConfigParser()
        self.project_config = config

    @property
    def get_project_config(self):
        """
        获取项目的配置信息
        :return:
        """
        self.project_config.read(f"{Path(__file__).resolve().parent}/config.ini")
        return self.project_config
