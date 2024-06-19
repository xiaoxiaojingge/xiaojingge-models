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
# json
import json
# 系统相关
import os
# 随机处理
import random
# 文件操作
import shutil
# 多线程相关
import threading
# 多进程相关
from multiprocessing import Queue

# 错误信息美化
import pretty_errors
# fastapi相关
from fastapi import File, UploadFile
# 深度学习工具sklearn
from sklearn.model_selection import train_test_split

# 日志服务
from config.logger import Logger
# 实体关系抽取模型服务
from models.task_relation_extraction_gplinker_bert4torch.model_server import \
    ModelServer

# 配置对象
config = configparser.ConfigParser()
# 日志对象
logger = Logger().get_logger


def split_train_data(
    input_file, train_file, val_file, test_file, train_ratio, val_ratio, if_test
):
    """
    将一个文件拆分成训练集、验证集、测试集
    :param if_test: 是否划分测试集
    :param val_ratio:  验证集生成比例
    :param train_ratio: 训练集生成比例
    :param input_file: 输入文件
    :param train_file: 训练集文件名称
    :param val_file: 验证集文件名称
    :param test_file: 测试集文件名称
    :return:
    """
    # 读取原始文件数据
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 打乱数据
    random.shuffle(lines)
    # 首先将数据划分为训练集和其他数据集
    train_data, other_data = train_test_split(lines, test_size=1 - train_ratio)
    # 是否划分测试数据集
    if if_test:
        # 将其他数据集划分为验证集和测试集
        val_size = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(
            other_data, test_size=val_size, random_state=42
        )
        # 保存拆分后的文件数据
        with open(train_file, "w", encoding="utf-8") as f:
            f.writelines(line.encode("utf-8").decode("utf-8") for line in train_data)
        with open(val_file, "w", encoding="utf-8") as f:
            f.writelines(line.encode("utf-8").decode("utf-8") for line in val_data)
        with open(test_file, "w", encoding="utf-8") as f:
            f.writelines(line.encode("utf-8").decode("utf-8") for line in test_data)
    else:
        # 保存拆分后的文件数据
        with open(train_file, "w", encoding="utf-8") as f:
            f.writelines(line.encode("utf-8").decode("utf-8") for line in train_data)
        with open(val_file, "w", encoding="utf-8") as f:
            f.writelines(line.encode("utf-8").decode("utf-8") for line in other_data)


def generate_all_schemas(source_file, target_file):
    """
    根据原本的标注文件，生成一个关系对应json，模型训练需要使用
    :param source_file:
    :param target_file:
    :return:
    """
    with open(source_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        handle_spo_list = []
        for line in lines:
            line_json = json.loads(line)
            spo_list = line_json["spo_list"]
            for spo in spo_list:
                object_type = spo["object_type"]
                predicate = spo["predicate"]
                subject_type = spo["subject_type"]
                handle_spo_list.append(
                    {
                        "object_type": object_type,
                        "predicate": predicate,
                        "subject_type": subject_type,
                    }
                )
        # 使用列表推导式和集合来去重字典
        unique_data = [dict(t) for t in {tuple(d.items()) for d in handle_spo_list}]
        with open(target_file, "w", encoding="utf-8") as file:
            for content in unique_data:
                json.dump(content, file, ensure_ascii=False)
                file.write("\n")


async def train_model(model_id: int, train_file: UploadFile = File(...)):
    """
    模型训练
    :param model_id: 模型id，模型管理的数据库id，用于区分本地模型信息
    :param train_file: 模型训练语料
    :return:
    """
    # 校验输入的模型输入文件是否是bio文件
    if not train_file.filename.endswith(".re"):
        return {
            "code": 400,
            "msg": "模型训练语料文件格式错误，请上传.re格式文件",
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
        model_train_dir = f"{train_data_dir}/re/{str(model_id)}/train"
        logger.info("开始执行模型训练逻辑......")
        logger.info(f"模型训练目录为 {model_train_dir}......")
        # 模型保存路径
        model_save_dir = f"{train_data_dir}/re/{str(model_id)}/models"
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
        # 判断模型保存目录的模型文件是否存在（best_model_gplinker.pt），如果存在则备份，后缀添加日期时间戳
        model_name = "best_model_gplinker.pt"
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
        with open(f"{model_train_dir}/all.re", "wb") as f:
            logger.info("开始保存模型训练语料......")
            f.write(train_data)

        split_train_data(
            model_train_dir + "/all.re",
            model_train_dir + "/train.re",
            model_train_dir + "/valid.re",
            model_train_dir + "/test.re",
            0.8,
            0.2,
            False,
        )
        # 生成一个关系对应文件，表明了数据中一共有多少种对应关系
        generate_all_schemas(
            f"{model_train_dir}/all.re", f"{model_train_dir}/all_schemas"
        )

        # 保存结果的队列
        result_queue = Queue()
        # 定义一个进程，传入参数和队列
        model_server = ModelServer(f"{train_data_dir}/re/{str(model_id)}", "train")
        t = threading.Thread(target=model_server.start_train, args=(result_queue,))
        # 开启进程，等待调度
        t.start()
        # 注意此操作会一直阻塞主线程，一直到队列里面有值为止，需要酌情设计，否则一直阻塞
        train_model_result = result_queue.get()
        # 等待进程获取结果
        t.join()
        return train_model_result
    except Exception as e:
        logger.error(f"模型训练异常，异常信息为：{e}")
        result = {
            "code": 500,
            "msg": f"模型训练异常，异常信息为：{e}",
            "data": {},
        }
        return result
