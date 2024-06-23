# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-23 13:48
@Author  : lijing
@File    : consumer.py
@Description: 消费者
---------------------------------------
"""
from kombu import Connection
from kombu.mixins import ConsumerMixin

from mq import redis_url, queue

import ast

from service.third_party.demo_service import train_model_logic

import threading


def process_model_train_task(body, message):
    """
    处理模型训练任务
    :param body:
    :param message:
    :return:
    """
    model_train_entity = ast.literal_eval(body)
    model_id = model_train_entity.get("model_id")
    workspace = model_train_entity.get("workspace")
    train_data_path = model_train_entity.get("train_data_path")
    user_account = model_train_entity.get("user_account")
    train_model_logic(model_id, workspace, train_data_path, user_account)
    message.ack()


class Worker(ConsumerMixin):
    def __init__(self, connection):
        self.connection = connection

    def get_consumers(self, Consumer, channel):
        # 创建消费者，指定回调函数
        return [
            Consumer(
                queues=[queue], accept=["json"], callbacks=[process_model_train_task]
            )
        ]


def run_consumer():
    """
    启动消费者
    :return:
    """
    with Connection(redis_url) as conn:
        worker = Worker(conn)
        worker.run()


if __name__ == "__main__":
    run_consumer()
