# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-23 13:47
@Author  : lijing
@File    : producer.py
@Description: 生产者
---------------------------------------
"""

from kombu import Connection, Producer

from mq import redis_url, exchange, queue


def publish_message(message):
    with Connection(redis_url) as conn:
        producer = Producer(conn)
        producer.publish(
            message,
            exchange=exchange,
            routing_key="model.train",
            declare=[queue],
            serializer="json",
        )
