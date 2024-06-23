# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 15:10
@Author  : lijing
@File    : main.py
@Description: 项目主入口文件
---------------------------------------
"""

# 配置相关
from config.config import Config

import uvicorn
from fastapi import FastAPI

# FastAPI 文档推荐使用 Uvicorn 来部署应用(其次是 hypercorn)
# Uvicorn 是一个基于 asyncio 开发的一个轻量级高效的 Web 服务器框架(仅支持 python 3.5.3 以上版本)
from api import model_api

# 日志打印类
from config.logger import Logger

# 数据库相关
import db

# redis
from utils.redis_util import RedisUtil

# 消费者
# from mq.consumer import run_consumer

app = FastAPI()
config = Config().get_project_config
logger = Logger().get_logger
redis_util = RedisUtil(host="192.168.0.201", db=15, password="123456")


if __name__ == "__main__":
    try:
        # 读取.ini文件
        # 获取配置信息
        config.read("config/config.ini")
        port = int(config.get("uvicorn", "port"))

        # 添加路由
        # 模型相关路由
        app.include_router(model_api.router)

        # 系统运行初始化模型是否训练为否
        # redis_util.set("model:train_status", "false")

        logger.info(f"本程序将在{port}端口运行......")

        # run_consumer()
        # logger.info("启动消息队列消费未训练完成的任务......")

        # 运行
        uvicorn.run(app=app, host="0.0.0.0", port=port, workers=1, log_level="error")
    except KeyboardInterrupt:
        logger.info("程序主程序运行被终止...")
