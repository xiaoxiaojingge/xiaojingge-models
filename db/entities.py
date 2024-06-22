# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-22 0:31
@Author  : lijing
@File    : entities.py
@Description: 实体类
---------------------------------------
"""

from sqlalchemy import Column, Integer, String, LargeBinary
from db import Base, engine


class ModelInfoEntity(Base):
    """
    模型实体
    """

    __tablename__ = "model_info"
    __table_args__ = {"comment": "模型信息"}

    id = Column("id", Integer, primary_key=True, autoincrement=True, comment="主键id")
    model_param = Column("model_param", String(50), nullable=True, comment="模型参数")
    model_value = Column("model_value", String(50), nullable=True, comment="模型值")


class ModelTrainEntity(Base):
    """
    模型训练实体，待训练的模型信息
    """

    __tablename__ = "model_train"
    __table_args__ = {"comment": "模型训练"}

    id = Column("id", Integer, primary_key=True, autoincrement=True, comment="主键id")
    model_id = Column("model_id", Integer, nullable=True, comment="模型id")
    user_account = Column("user_account", String(50), nullable=True, comment="用户账户")
    train_data_path = Column(
        "train_data_path", String(200), nullable=True, comment="训练数据语料本地地址"
    )


# 数据库未创建表的话自动创建表
Base.metadata.create_all(engine)
