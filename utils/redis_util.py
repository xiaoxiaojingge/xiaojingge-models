# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-23 19:13
@Author  : lijing
@File    : redis_util.py
@Description: redis工具类
---------------------------------------
"""
import redis
from typing import Any, List, Dict
import configparser

# 配置相关
from config.config import Config

config = Config().get_project_config


class RedisUtil:
    def __init__(self, host="localhost", port=6379, db=0, password=None):
        """
        初始化Redis工具类

        :param host: Redis服务器的主机名，默认为'localhost'
        :param port: Redis服务器的端口号，默认为6379
        :param db: 使用的Redis数据库编号，默认为0
        :param password: 连接Redis服务器的密码，默认为None
        """
        self.pool = redis.ConnectionPool(host=host, port=port, db=db, password=password)
        self.client = redis.Redis(connection_pool=self.pool)

    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """
        设置键值对

        :param key: 键名
        :param value: 键值
        :param ex: 过期时间（秒），默认为None
        :return: 操作是否成功
        """
        return self.client.set(key, value, ex=ex)

    def get(self, key: str) -> Any:
        """
        获取键对应的值

        :param key: 键名
        :return: 键值
        """
        return self.client.get(key)

    def delete(self, key: str) -> int:
        """
        删除指定键

        :param key: 键名
        :return: 删除的键数量
        """
        return self.client.delete(key)

    def exists(self, key: str) -> bool:
        """
        检查键是否存在

        :param key: 键名
        :return: 键是否存在
        """
        return self.client.exists(key) > 0

    def expire(self, key: str, time: int) -> bool:
        """
        设置键的过期时间

        :param key: 键名
        :param time: 过期时间（秒）
        :return: 操作是否成功
        """
        return self.client.expire(key, time)

    def ttl(self, key: str) -> int:
        """
        获取键的剩余生存时间

        :param key: 键名
        :return: 剩余生存时间（秒）
        """
        return self.client.ttl(key)

    def keys(self, pattern: str = "*") -> List[str]:
        """
        查找所有符合给定模式(pattern)的键

        :param pattern: 模式，默认为'*'（匹配所有键）
        :return: 匹配的键列表
        """
        return self.client.keys(pattern)

    def hset(self, name: str, key: str, value: Any) -> int:
        """
        设置哈希表中的字段值

        :param name: 哈希表名
        :param key: 字段名
        :param value: 字段值
        :return: 被设置字段的数量
        """
        return self.client.hset(name, key, value)

    def hget(self, name: str, key: str) -> Any:
        """
        获取哈希表中的字段值

        :param name: 哈希表名
        :param key: 字段名
        :return: 字段值
        """
        return self.client.hget(name, key)

    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        获取哈希表中的所有字段和值

        :param name: 哈希表名
        :return: 哈希表中的字段和值
        """
        return self.client.hgetall(name)

    def lpush(self, name: str, *values) -> int:
        """
        将一个或多个值插入到列表头部

        :param name: 列表名
        :param values: 值
        :return: 列表的长度
        """
        return self.client.lpush(name, *values)

    def rpush(self, name: str, *values) -> int:
        """
        将一个或多个值插入到列表尾部

        :param name: 列表名
        :param values: 值
        :return: 列表的长度
        """
        return self.client.rpush(name, *values)

    def lpop(self, name: str) -> Any:
        """
        移除并返回列表的头元素

        :param name: 列表名
        :return: 头元素
        """
        return self.client.lpop(name)

    def rpop(self, name: str) -> Any:
        """
        移除并返回列表的尾元素

        :param name: 列表名
        :return: 尾元素
        """
        return self.client.rpop(name)

    def sadd(self, name: str, *values) -> int:
        """
        向集合添加一个或多个成员

        :param name: 集合名
        :param values: 成员
        :return: 添加到集合中的新成员的数量
        """
        return self.client.sadd(name, *values)

    def srem(self, name: str, *values) -> int:
        """
        移除集合中一个或多个成员

        :param name: 集合名
        :param values: 成员
        :return: 从集合中移除的成员的数量
        """
        return self.client.srem(name, *values)

    def smembers(self, name: str) -> set:
        """
        返回集合中的所有成员

        :param name: 集合名
        :return: 集合中的所有成员
        """
        return self.client.smembers(name)

    def zadd(self, name: str, mapping: Dict[str, float]) -> int:
        """
        向有序集合添加一个或多个成员，或者更新已存在成员的分数

        :param name: 有序集合名
        :param mapping: 字典形式的成员和分数
        :return: 新添加到有序集合的成员数量
        """
        return self.client.zadd(name, mapping)

    def zrange(
        self,
        name: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
    ) -> List:
        """
        返回有序集中，指定区间内的成员

        :param name: 有序集合名
        :param start: 起始位置
        :param end: 结束位置
        :param desc: 是否按分数从高到低排序，默认为False
        :param withscores: 是否返回成员的分数，默认为False
        :return: 指定区间内的成员
        """
        return self.client.zrange(name, start, end, desc=desc, withscores=withscores)

    def acquire_lock(self, lock_name: str, timeout: int) -> bool:
        """
        获取锁

        :param lock_name: 锁的名称
        :param timeout: 锁的超时时间（秒）
        :return: 是否成功获取锁
        """
        return self.client.set(lock_name, "locked", ex=timeout, nx=True)

    def release_lock(self, lock_name: str) -> bool:
        """
        释放锁

        :param lock_name: 锁的名称
        :return: 是否成功释放锁
        """
        return bool(self.client.delete(lock_name))


# redis_util对象
redis_util = RedisUtil(
    host=config.get("redis", "host"), password=config.get("redis", "password"), db=15
)


if __name__ == "__main__":
    # 加锁操作
    redis_util.set("train_status", "false")
    # lock_name = "train_lock"
    # # 尝试获取锁，超时时间设为60秒
    # if redis_util.acquire_lock(lock_name, 60):
    #     try:
    #         # 检查 train_status 是否为 false
    #         if redis_util.get("train_status") == b"false":
    #             redis_util.set("train_status", "true")
    #         else:
    #             pass
    #     finally:
    #         redis_util.release_lock(lock_name)
