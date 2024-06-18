# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 16:57
@Author  : lijing
@File    : model_api.py
@Description: 模型相关接口
---------------------------------------
"""

# web api 框架
# from fastapi import FastAPI
import json

from fastapi import APIRouter, File, Form, UploadFile

# 日志打印类
from config.logger import logger

# 模型类型常量
from constants import model_constants

# 业务操作
# TODO

# 路由实例
router = APIRouter()


@router.post("/test")
async def test(param: str = Form(...), file: UploadFile = File(...)):
    result = {"code": 200, "message": "success", "data": {}}
    return result
