# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-22 0:31
@Author  : lijing
@File    : entities.py
@Description: 实体类
---------------------------------------
"""

from sqlalchemy import Column, Integer, String, TIMESTAMP, func
from db import Base, engine


class ModelTrainEntity(Base):
    """
    模型训练实体，待训练的模型信息
    """0

    __tablename__ = "model_train"
    __table_args__ = {"comment": "模型训练"}

    id = Column("id", Integer, primary_key=True, autoincrement=True, comment="主键id")
    model_id = Column("model_id", Integer, nullable=True, comment="模型id")
    user_account = Column("user_account", String(50), nullable=True, comment="用户账户")
    workspace = Column("workspace", String(200), nullable=True, comment="工作空间")
    train_data_path = Column(
        "train_data_path", String(200), nullable=True, comment="训练数据语料本地地址"
    )
    module = Column("module", String(200), nullable=True, comment="业务模块")
    remote_module = Column(
        "remote_module", String(200), nullable=True, comment="远程模块"
    )
    train_result = Column(
        "train_result", String(200), nullable=True, comment="训练结果"
    )
    create_time = Column(
        "create_time",
        TIMESTAMP,
        nullable=True,
        server_default=func.now(),
        comment="创建时间",
    )


# 数据库未创建表的话自动创建表
Base.metadata.create_all(engine)
