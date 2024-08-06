# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-23 13:46
@Author  : lijing
@File    : __init__.py
@Description: 
---------------------------------------
"""
from kombu import Exchange, Queue

# 配置相关
from config.config import Config

config = Config().get_project_config

# 定义连接字符串，使用Redis作为消息队列
redis_url = config.get("redis", "url")

# 定义交换机和队列
exchange = Exchange("model_train_exchange", type="direct", durable=True)
queue = Queue(
    "model_train_queue", exchange=exchange, routing_key="model.train", durable=True
)
