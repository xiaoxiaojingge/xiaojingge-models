# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 16:57
@Author  : lijing
@File    : model_api.py
@Description: 模型相关接口
---------------------------------------
"""
import time

from fastapi import APIRouter, File, Form, UploadFile

# 业务操作
# from service import *


# 路由实例
router = APIRouter()


@router.post("/test")
def test(modelId: int = Form(...), file: UploadFile = File(...)):
    # result = demo_service.train_model_re(modelId, "lijing", file)
    result = []
    print(111)
    time.sleep(10)
    return result
