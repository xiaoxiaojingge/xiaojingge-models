# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-20 10:00
@Author  : lijing
@File    : CustomException.py
@Description: 自定义异常
---------------------------------------
"""


class BizException(Exception):
    """
    自定义业务异常
    """

    def __init__(self, message):
        self.message = message
