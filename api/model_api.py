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

from fastapi import APIRouter, File, Form, UploadFile

# 业务操作
# 命名实体识别业务
import service.task_relation_extraction_gplinker_bert4torch.model_service as re_service

# 日志打印类

# 模型类型常量

# TODO

# 路由实例
router = APIRouter()


@router.post("/test")
async def test(param: str = Form(...), file: UploadFile = File(...)):
    result = await re_service.predict(model_id=2, predict_file=file)
    return result
