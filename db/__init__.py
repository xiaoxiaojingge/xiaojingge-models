# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-22 0:17
@Author  : lijing
@File    : __init__.py
@Description: 
---------------------------------------
"""
# 数据库相关，sqlalchemy为orm的底层实现
from sqlalchemy import create_engine, event
from sqlalchemy.exc import DisconnectionError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

# 配置相关
from config.config import Config

# 错误信息美化
import pretty_errors

# 日志
from config.logger import Logger

logger = Logger().get_logger

config = Config().get_project_config

mysql_url = config.get("mysql", "url")


# 监听函数，用于在连接检出期间检查数据库连接的健康状况。
def checkout_listener(dbapi_con, con_record, con_proxy):
    try:
        try:
            dbapi_con.ping(False)
        except TypeError:
            dbapi_con.ping()
    except dbapi_con.OperationalError as exc:
        if exc.args[0] in (2006, 2013, 2014, 2045, 2055):
            raise DisconnectionError()
        else:
            raise RuntimeError(f"数据库连接检出期间发生错误: {exc}")


# 检查mysql的url是否设置
if not mysql_url:
    logger.error("mysql url is not set")
    raise Exception("mysql url is not set")


# 创建SQLAlchemy引擎
def get_engine(echo: bool = False) -> Engine:

    engine_info = create_engine(
        mysql_url,
        echo=True,
        pool_pre_ping=True,
        pool_size=100,
        pool_recycle=360,
    )

    # 在SQLAlchemy中使用事件系统来记录SQL语句
    # @event.listens_for(engine_info, "before_cursor_execute")
    # def before_cursor_execute(conn, cursor, statement, parameters, context, executor):
    #     # 在执行SQL之前记录
    #     logger.info(f"SQL Statement: {statement}")
    #     logger.info(f"Parameters: {parameters}")
    #
    # @event.listens_for(engine_info, "after_cursor_execute")
    # def after_cursor_execute(conn, cursor, statement, parameters, context, executor):
    #     # 在执行SQL之后记录
    #     logger.info(f"SQL Result: {cursor.rowcount}")

    return engine_info


engine = get_engine()

event.listen(engine, "checkout", checkout_listener)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


# 数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
