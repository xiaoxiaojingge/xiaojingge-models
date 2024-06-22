# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-21 11:30
@Author  : lijing
@File    : demo_service.py
@Description: demo业务
---------------------------------------
"""
# 系统相关
import os

# http请求库
import requests

# fastapi相关
from fastapi import File, UploadFile

# 数据库相关
import db
from db.entities import *

# 配置相关
from config.config import Config

# 日志打印类
from config.logger import Logger

# 自定义异常
from exception.CustomException import BizException

# 关系实体模型相关
import service.task_relation_extraction_gplinker_bert4torch.model_service as re_service

# 异步相关
import asyncio

# 时间相关
import datetime

# 多线程相关
import threading

config = Config().get_project_config
logger = Logger().get_logger


def train_model_re(
    model_id: int, user_account: str, train_file: UploadFile = File(...)
):
    """
    关系实体模型训练
    :param model_id: 模型id
    :param user_account: 用户名
    :param train_file: 训练文件
    :return:
    """
    session = next(db.get_db())
    try:
        valid_empty_message = "模型训练语料文件为空，请上传非空文件"
        valid_format_message = "模型训练语料文件格式错误，请上传.re格式文件"
        if not train_file:
            logger.error(valid_empty_message)
            raise BizException(valid_empty_message)

        if not train_file.filename.endswith(".re"):
            logger.error(valid_format_message)
            raise BizException(valid_format_message)
        train_data = train_file.file.read()

        if len(train_data) == 0:
            logger.error(valid_empty_message)
            raise BizException(valid_empty_message)

        # 将训练语料保存到本地
        model_train_data_save_dir = f"{config.get('model_train', 'model_train_data_save_dir')}/{user_account}/{datetime.datetime.now().strftime('%Y-%m-%d')}/{model_id}"
        if not os.path.exists(model_train_data_save_dir):
            os.makedirs(model_train_data_save_dir)
        train_data_path = f"{model_train_data_save_dir}/{train_file.filename}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        with open(
            train_data_path,
            "wb",
        ) as f:
            # 读取上传文件的内容，并将其写入到本地文件
            f.write(train_data)

        # 修改模型训练状态
        sys_model_info = (
            session.query(ModelInfoEntity)
            .filter(ModelInfoEntity.model_param == "is_train")
            .first()
        )

        # 如果服务器资源被占用
        if sys_model_info.model_value == "True":
            model_train = ModelTrainEntity(
                model_id=model_id,
                user_account=user_account,
                train_data_path=train_data_path,
            )
            # 保存训练记录到数据库
            session.add(model_train)
            session.commit()
            return {
                "code": 200,
                "message": "模型训练任务提交成功，请稍后查看模型训练状态！",
                "data": {},
            }

        # 异步训练
        thread = threading.Thread(
            target=train_model_logic,
            args=(model_id, sys_model_info, train_data_path, user_account, session),
        )
        thread.start()
        return {
            "code": 200,
            "message": "模型训练任务提交成功，请稍后查看模型训练状态！",
            "data": {},
        }
    except Exception as e:
        logger.error(f"执行模型训练逻辑发生异常，异常信息：{e}")
        session.rollback()


def train_model_logic(model_id, sys_model_info, train_data_path, user_account, session):
    """
    模型训练逻辑
    :param model_id: 模型id
    :param sys_model_info: 模型训练状态实体
    :param train_data_path: 训练语料路径
    :param user_account: 用户名
    :param session:
    :return:
    """
    # 模型训练中
    sys_model_info.model_value = "True"
    session.merge(sys_model_info)
    session.commit()
    model_train = ModelTrainEntity(
        model_id=model_id,
        user_account=user_account,
        train_data_path=train_data_path,
    )
    # 保存训练记录到数据库
    session.add(model_train)
    session.commit()
    # 开始训练
    train_data_dir = config.get("model_train", "model_train_dir")
    workspace = f"{train_data_dir}/{user_account}/{datetime.datetime.now().strftime('%Y-%m-%d')}/{model_id}"
    result = re_service.train_model(workspace, train_data_path)
    if result["code"] == 500:
        raise BizException(result["message"])
    # 完成后，删除训练记录
    session.delete(model_train)
    session.commit()
    # 回调远程服务，将训练结果保存
    save_train_model_result(result)
    # 模型未训练
    sys_model_info.model_value = "False"
    session.merge(sys_model_info)
    session.commit()


def predict_re(model_id: int, user_account: str, predict_file: UploadFile = File(...)):
    """
    关系实体模型预测
    :param model_id: 模型id
    :param user_account: 用户名
    :param predict_file: 预测文件
    :return:
    """
    train_data_dir = config.get("model_train", "model_train_dir")
    workspace = f"{train_data_dir}/{user_account}/{datetime.datetime.now().strftime('%Y-%m-%d')}/{model_id}"
    result = re_service.predict(workspace, predict_file)
    logger.success(f"工作目录为 {workspace} 的模型任务已完成！")
    return result


def save_train_model_result(result):
    """
    保存模型训练结果
    :param result:
    :return:
    """
    # 获取demo系统服务地址
    demo_server_url = config.get("third_party", "demo_server_url")
    # 调用保存模型训练结果接口，保存该次模型训练的结果
    logger.info(f"demo系统服务地址：{demo_server_url}")
    # 请求头
    headers = {
        "Content-Type": "application/json",
    }
    demo_server_url = "https://myhkw.cn/open/ip"
    response = requests.post(demo_server_url, data=result, headers=headers)
    logger.info(f"远程调用结果 -----> {response.json()}")
    success_message = "回调函数 -----> 保存模型训练结果成功！"
    fail_message = "回调函数 -----> 保存模型训练结果失败！"
    if response.status_code != 200:
        logger.error(fail_message)
        raise BizException(fail_message)
    else:
        if response.json()["code"] != 0:
            logger.error(fail_message)
        else:
            logger.info(success_message)


if __name__ == "__main__":
    pass
