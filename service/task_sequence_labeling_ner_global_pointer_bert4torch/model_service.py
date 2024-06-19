# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-19 9:35
@Author  : lijing
@File    : model_service.py
@Description:  模型相关业务类
---------------------------------------
"""

# 配置相关
import configparser
# 日期相关
import datetime
# 系统相关
import os
# 文件操作
import shutil

# fastapi相关
from fastapi import File, UploadFile

# 日志服务
from config.logger import Logger
# 命名实体识别模型服务
from models.task_sequence_labeling_ner_global_pointer_bert4torch.model_server import \
    ModelServer

# 配置对象
config = configparser.ConfigParser()
# 日志对象
logger = Logger().get_logger


async def train_model(model_id: int, train_file: UploadFile = File(...)):
    """
    模型训练
    :param model_id: 模型id，模型管理的数据库id，用于区分本地模型信息
    :param train_file: 模型训练语料
    :return:
    """
    # 校验输入的模型输入文件是否是bio文件
    if not train_file.filename.endswith(".bio"):
        return {
            "code": 400,
            "msg": "模型训练语料文件格式错误，请上传.bio格式文件",
            "data": {},
        }
    train_data = await train_file.read()

    if len(train_data) == 0:
        return {
            "code": 400,
            "msg": "模型训练语料文件为空，请上传非空文件",
            "data": {},
        }
    # 模型的语料数量
    train_data_count = 0
    try:
        # 读取.ini文件
        config.read("config/config.ini")
        train_data_dir = config.get("model_train", "model_train_dir")
        # 模型训练路径
        model_train_dir = f"{train_data_dir}/ner/{str(model_id)}/train"
        logger.info("开始执行模型训练逻辑......")
        logger.info(f"模型训练目录为 {model_train_dir}......")
        # 模型保存路径
        model_save_dir = f"{train_data_dir}/ner/{str(model_id)}/models"
        logger.info(f"模型产物保存目录为 {model_save_dir}......")
        # 检查训练目录是否存在
        if not os.path.exists(model_train_dir):
            # 创建目录
            os.makedirs(model_train_dir)
        else:
            # 清空训练目录的文件数据
            shutil.rmtree(model_train_dir)
            os.makedirs(model_train_dir)
        # 判断模型保存目录是否存在
        if not os.path.exists(model_save_dir):
            # 创建目录
            os.makedirs(model_save_dir)
        # 判断模型保存目录的模型文件是否存在（best_model.weights），如果存在则备份，后缀添加日期时间戳
        model_name = "best_model.weights"
        if os.path.exists(f"{model_save_dir}/{model_name}"):
            logger.info("模型产物目录已经存在模型文件，开始备份模型文件......")
            # 构造备份文件名，加上时间戳作为后缀
            # 获取当前时间作为时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file_name = f"{model_name}.{timestamp}"
            backup_file_path = os.path.join(model_save_dir, backup_file_name)
            shutil.copy(f"{model_save_dir}/{model_name}", backup_file_path)
            os.remove(f"{model_save_dir}/{model_name}")

        #  将上传的压缩包保存到本地目录
        with open(f"{model_train_dir}/all.bio", "wb") as f:
            logger.info("开始保存模型训练语料......")
            f.write(train_data)
        # 划分数据集，按照比例，将数据划分为训练集、验证集、测试集

    except Exception as e:
        pass
