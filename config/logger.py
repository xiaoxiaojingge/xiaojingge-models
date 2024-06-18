#! -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2023-09-18
@Author  : lijing
@File    : logger.py
@Description: 强大的日志记录库loguru
---------------------------------------
'''

from loguru import logger

logger.add(
    "logs/app.log",     # 日志存放路径
    rotation="12:00",  # 每天12:00会创建一个新的文件
    level="DEBUG",     # 日志等级
    format="[{time:YYYY-MM-DD HH:mm:ss}] - {module}:{line} - {level} - {message}",  # 日志输出格式
    enqueue=True,  # 多进程/异步记录安全
    backtrace=True,  # 打印错误堆栈
    diagnose=True
)