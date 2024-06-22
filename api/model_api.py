# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 16:57
@Author  : lijing
@File    : model_api.py
@Description: 模型相关接口
---------------------------------------
"""

from fastapi import APIRouter, File, Form, UploadFile

# 业务操作
from service import *


# 路由实例
router = APIRouter()


@router.post("/test")
def test(param: str = Form(...), file: UploadFile = File(...)):
    result = demo_service.train_model_re(2, "lijing", file)
    return result
