#! -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2023-09-18
@Author  : lijing
@File    : logger.py
@Description: 强大的日志记录库loguru
---------------------------------------
"""
import datetime
import os
from functools import wraps
# 路径相关
from pathlib import Path

import loguru


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
class Logger:
    def __init__(self):
        self.logger_add()

    def get_project_path(self, project_path=None):
        if project_path is None:
            project_path = str(Path(__file__).resolve())
        # 返回当前项目路径
        return project_path

    def get_log_path(self):
        # 项目目录
        project_path = self.get_project_path(Path(__file__).resolve().parent.parent)
        # 项目日志目录
        project_log_dir = os.path.join(project_path, "logs")
        # 返回日志路径
        return project_log_dir

    def logger_add(self):
        loguru.logger.add(
            sink=os.path.join(
                self.get_log_path(), "app_{}.log".format(datetime.date.today())
            ),
            # 日志创建周期
            rotation="00:00",
            # 保存
            retention="1 year",
            # 日志等级
            level="DEBUG",
            # 文件的压缩格式
            compression="zip",
            # 编码格式
            encoding="utf-8",
            # 具有使日志记录调用非阻塞的优点，多进程/异步记录安全
            enqueue=True,
            # 日志输出格式
            format="[{time:YYYY-MM-DD HH:mm:ss}] - {module}:{line} - {level} - {message}",
            # 打印错误堆栈
            backtrace=True,
            diagnose=True,
        )
        loguru.logger.add(
            sink=os.path.join(
                self.get_log_path(), "app_error_{}.log".format(datetime.date.today())
            ),
            # 日志创建周期
            rotation="00:00",
            # 保存
            retention="1 year",
            # 日志等级
            level="ERROR",
            # 文件的压缩格式
            compression="zip",
            # 编码格式
            encoding="utf-8",
            # 具有使日志记录调用非阻塞的优点，多进程/异步记录安全
            enqueue=True,
            # 日志输出格式
            format="[{time:YYYY-MM-DD HH:mm:ss}] - {module}:{line} - {level} - {message}",
            # 打印错误堆栈
            backtrace=True,
            diagnose=True,
        )

    # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加（）。
    @property
    def get_logger(self):
        return loguru.logger
