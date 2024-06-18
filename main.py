# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 15:10
@Author  : lijing
@File    : main.py
@Description: 项目主入口文件
---------------------------------------
"""

# FastAPI 文档推荐使用 Uvicorn 来部署应用(其次是 hypercorn)
# Uvicorn 是一个基于 asyncio 开发的一个轻量级高效的 Web 服务器框架(仅支持 python 3.5.3 以上版本)
from api import model_api
from fastapi import FastAPI
import uvicorn
import configparser

# 日志打印类
from config.logger import logger

# 打印工具类
import utils.print_util as print_util

app = FastAPI()
config = configparser.ConfigParser()

if __name__ == "__main__":
    # 读取.ini文件
    config.read("config/config.ini")
    # 获取配置信息
    port = int(config.get("uvicorn", "port"))
    # 添加路由
    # 模型相关路由
    app.include_router(model_api.router)
    logger.info(f"本程序将在{port}端口运行......")
    # 运行
    uvicorn.run(app=app, host="0.0.0.0", port=port, workers=1, log_level="error")
