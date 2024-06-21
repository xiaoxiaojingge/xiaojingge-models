# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-22 0:31
@Author  : lijing
@File    : entities.py
@Description: 实体类
---------------------------------------
"""

from sqlalchemy import Column, Integer, Boolean, TIMESTAMP, func
from db import Base, engine


class SysModelEntity(Base):
    """系统模型实体"""

    __tablename__ = "sys_model"
    __table_args__ = {"comment": "系统模型"}

    id = Column("id", Integer, primary_key=True, autoincrement=True, comment="主键id")
    nickname = Column("model_status", Boolean, nullable=True, comment="模型状态")
    create_time = Column(
        "create_time", TIMESTAMP, server_default=func.now(), comment="创建时间"
    )
    update_time = Column(
        "update_time", TIMESTAMP, server_default=func.now(), comment="修改时间"
    )


# 数据库未创建表的话自动创建表
Base.metadata.create_all(engine)
