# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-17 15:10
@Author  : lijing
@File    : main.py
@Description: 项目主入口文件
---------------------------------------
"""

# 配置相关
from config.config import Config

import uvicorn
from fastapi import FastAPI

# FastAPI 文档推荐使用 Uvicorn 来部署应用(其次是 hypercorn)
# Uvicorn 是一个基于 asyncio 开发的一个轻量级高效的 Web 服务器框架(仅支持 python 3.5.3 以上版本)
from api import model_api

# 日志打印类
from config.logger import Logger

# 模块导入
import importlib

# 定时任务
from schedule.schedule import Schedule

# 异常
from exception.CustomException import *

app = FastAPI()
config = Config().get_project_config
logger = Logger().get_logger


def run_train_tasks():
    """
    定时任务拉起数据库中未训练完成的任务
    :return:
    """
    # 定时任务检查数据库训练任务
    logger.info("定时任务检查数据库训练任务中...")
    with next(db.get_db()) as session:
        try:
            model_train_entity = (
                session.query(ModelTrainEntity)
                .filter(ModelTrainEntity.if_delete == False)
                .order_by(ModelTrainEntity.id)
                .first()
            )

            if model_train_entity is None:
                return
            logger.info("检测到有模型训练任务，服务器资源信息检验中......")
            sys_model_info = (
                session.query(ModelInfoEntity)
                .filter(ModelInfoEntity.model_param == "is_train")
                .filter(ModelInfoEntity.model_value == "False")
                .first()
            )
            if sys_model_info is None:
                logger.info(
                    "当前有模型任务正在训练中，本次任务后续将自动加入训练队列..."
                )
                return

            # 获取当前版本号
            current_version = sys_model_info.version
            try:
                # 修改 sys_model_info 的值并更新版本号
                # 使用 where 子句确保版本号未变
                update_stmt = (
                    ModelInfoEntity.__table__.update()
                    .where(ModelInfoEntity.id == sys_model_info.id)
                    .where(ModelInfoEntity.version == current_version)
                    .values(model_value="True", version=current_version + 1)
                )
                result = session.execute(update_stmt)

                if result.rowcount == 0:
                    raise BizException("数据已被其他事务修改！")

                session.commit()
            except Exception as e:
                logger.warning(f"检测到并发修改，任务放弃或重试: {e}")
                session.rollback()
                return

            service = importlib.import_module(model_train_entity.module)
            result = service.train_model(
                model_train_entity.workspace, model_train_entity.train_data_path
            )
            if result["code"] == 500:
                logger.error(f"执行模型训练逻辑发生异常，异常信息：{result['message']}")
                # raise BizException(result["message"])
            # 完成后，删除训练记录
            model_train_entity.if_delete = True
            model_train_entity.train_result = str(result)
            session.merge(model_train_entity)
            session.commit()
            # 回调远程服务，将训练结果保存
            remote_service = importlib.import_module(model_train_entity.remote_module)
            remote_service.save_train_model_result(result)
            # 模型未训练
            sys_model_info.model_value = "False"
            session.merge(sys_model_info)
            session.commit()
        except Exception as e:
            logger.info(f"定时任务拉起的模型训练任务运行出现异常：{e}")
            session.rollback()


if __name__ == "__main__":
    # 拉起数据库中未训练完成的任务
    schedule = Schedule(run_train_tasks, 3)
    try:
        # 读取.ini文件
        config.read("config/config.ini")  # 获取配置信息
        port = int(config.get("uvicorn", "port"))
        # 添加路由
        # 模型相关路由
        app.include_router(model_api.router)
        logger.info(f"本程序将在{port}端口运行......")
        # 系统运行初始化模型是否训练为否
        # 数据库相关
        import db
        from db.entities import *

        with next(db.get_db()) as session:
            sys_model_info = (
                session.query(ModelInfoEntity)
                .filter(ModelInfoEntity.model_param == "is_train")
                .first()
            )
            if sys_model_info is None:
                sys_model_info = ModelInfoEntity(
                    model_param="is_train", model_value="False"
                )
                session.add(sys_model_info)
            else:
                sys_model_info.model_value = "False"
            session.commit()
            session.close()

        schedule.start_loop_async()
        logger.info("定时任务正在拉起数据库中未训练完成的任务......")
        # 运行
        uvicorn.run(app=app, host="0.0.0.0", port=port, workers=1, log_level="error")
    except KeyboardInterrupt:
        logger.info("程序主程序运行被终止...")
        # 程序停止后停止所有定时任务
        schedule.stop_loop()
    except Exception as e:
        logger.error(f"程序主程序运行发生异常: {e}")
