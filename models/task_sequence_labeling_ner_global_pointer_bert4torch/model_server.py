# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-18 15:44
@Author  : lijing
@File    : model_server.py
@Description: 命名实体识别模型（task_sequence_labeling_ner_global_pointer_bert4torch）
---------------------------------------
"""

# 配置相关
import configparser
# json相关
import json
# 模型相关
# 系统相关
import os
# 多线程相关
from multiprocessing import Queue
# 路径相关
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.layers import GlobalPointer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.snippets import ListDataset, seed_everything, sequence_padding
from bert4torch.tokenizers import Tokenizer
from torch.utils.data import DataLoader

# 日志相关
from config.logger import logger
# 邮件相关
from utils.email_util import EmailServer


class MyDataset(ListDataset):
    """
    加载数据集
    """

    @staticmethod
    def load_data(filename):
        data = []
        with open(filename, encoding="utf-8") as f:
            f = f.read()
            for l in f.split("\n\n"):
                if not l:
                    continue
                text, label = "", []
                for i, c in enumerate(l.split("\n")):
                    char, flag = c.split(" ")
                    text += char
                    if flag[0] == "B":
                        label.append([i, i, flag[2:]])
                    elif flag[0] == "I":
                        label[-1][1] = i
                data.append((text, label))  # label为[[start, end, entity], ...]
        return data


class Model(BaseModel):
    """
    定义bert上的模型结构
    """

    def __init__(self, params: dict):
        super().__init__()
        self.bert = build_transformer_model(
            config_path=params["bert_config_path"],
            checkpoint_path=params["bert_checkpoint_path"],
            segment_vocab_size=0,
        )
        self.global_pointer = GlobalPointer(
            hidden_size=768,
            heads=params["ner_vocab_size"],
            head_size=params["ner_head_size"],
        )

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())
        return logit


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(
            y_true.shape[0] * y_true.shape[1], -1
        )  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(
            y_pred.shape[0] * y_pred.shape[1], -1
        )  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


class Evaluator(Callback):
    """评估与保存"""

    def __init__(self, model_server):
        self.best_val_f1 = 0.0
        self.model_server = model_server

    def on_epoch_end(self, steps, epoch, logs=None):
        global final_f1, final_precision, final_recall
        f1, precision, recall = self.model_server.evaluate(
            self.model_server.valid_dataloader
        )
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            # 保存更优f1的模型
            self.model_server.model.save_weights(
                self.model_server.model_dir + "/models/best_model.weights"
            )
            final_f1 = round(f1, 4)
            final_precision = round(precision, 4)
            final_recall = round(recall, 4)
        logger.info(
            "f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n"
            % (f1, precision, recall, self.best_val_f1)
        )


class ModelServer:
    """
    模型训练类
    """

    def __init__(self, model_dir: str, sign: str):
        config = configparser.ConfigParser()
        # 读取.ini文件，这里的文件使用相对路径拼接
        config.read(
            str(Path(__file__).resolve().parent.parent.parent) + "\config\config.ini"
        )
        # 获取bert模型相关信息
        bert_model_path = config.get("bert", "model_path")

        self.model_dir = model_dir

        # 邮件服务
        self.email_server = EmailServer(
            config.get("email", "smtp_server"),
            config.get("email", "smtp_port"),
            config.get("email", "sender_email"),
            config.get("email", "sender_password"),
            config.get("email", "receiver_email"),
        )

        # 模型参数信息
        # 表示训练的最大长度。这个参数决定了输入序列的最大长度，超过这个长度的序列将被截断，而短于这个长度的序列将被填充。
        self.max_len = 256
        # 表示训练的批次大小。这个参数决定了每次训练的样本数量。
        self.batch_size = 16
        # 表示训练的轮数。这个参数决定了模型在训练数据上进行训练的次数。
        self.epochs = 1
        # 表示训练的学习率。这个参数决定了模型在训练过程中学习率如何调整，以控制模型的训练速度和精度。
        self.learning_rate = 2e-5

        # bert配置
        # 表示 BERT 模型的配置文件位置。该文件通常是一个 JSON 格式的文件，其中包含了 BERT 模型的各种配置参数，例如模型的层数、隐藏单元数、注意力头数等。配置文件提供了对模型结构进行修改和定制的方式。
        self.bert_config_path = (
            bert_model_path + "/chinese_L-12_H-768_A-12/bert4torch_config.json"
        )
        # 表示 BERT 模型的检查点位置。检查点文件通常是训练过程中保存的模型参数的二进制文件。通过加载这个文件，可以恢复训练过程中的模型参数，或者用于对未知数据进行预测。
        self.bert_checkpoint_path = (
            bert_model_path + "/chinese_L-12_H-768_A-12/pytorch_model.bin"
        )
        # 表示 BERT 模型的字典文件位置。字典文件是一个文本文件，其中包含了模型所使用的词汇表信息。BERT 模型将输入的文本转化为对应的词向量表示，字典文件定义了模型所使用的词汇和对应的编码方式。
        self.bert_dict_path = bert_model_path + "/chinese_L-12_H-768_A-12/vocab.txt"

        # 初始化其他实例变量
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 固定seed
        seed_everything(42)

        # 建立分词器
        self.tokenizer = Tokenizer(self.bert_dict_path, do_lower_case=True)

        categories_label2id = self.get_entity_dict(model_dir + "/train/train.ner")
        categories_id2label = dict(
            (value, key) for key, value in categories_label2id.items()
        )
        ner_vocab_size = len(categories_label2id)
        ner_head_size = 64
        self.categories_label2id = categories_label2id
        self.categories_id2label = categories_id2label
        self.ner_vocab_size = ner_vocab_size
        self.ner_head_size = ner_head_size

        # 封装常用参数传递
        self.params = {
            "model_dir": self.model_dir,
            "max_len": self.max_len,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "bert_config_path": self.bert_config_path,
            "bert_checkpoint_path": self.bert_checkpoint_path,
            "bert_dict_path": self.bert_dict_path,
            "categories_label2id": self.categories_label2id,
            "categories_id2label": self.categories_id2label,
            "ner_vocab_size": self.ner_vocab_size,
            "ner_head_size": self.ner_head_size,
            "tokenizer": self.tokenizer,
        }

        self.model = Model(self.params).to(self.device)

        # 标注数据
        # 转换数据集
        # 训练集
        if sign == "train":
            self.train_dataloader = DataLoader(
                MyDataset(model_dir + "/train/train.ner"),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            # 验证集
            self.valid_dataloader = DataLoader(
                MyDataset(model_dir + "/train/valid.ner"),
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )

        self.model.compile(
            loss=MyLoss(),
            optimizer=optim.Adam(self.model.parameters(), lr=self.learning_rate),
        )

    def get_entity_dict(self, file_name, split_=" "):
        """
        生成所有实体的label2id，从0开始
        """
        # 读取一个文件（使用指定的文件名）并将其内容解析成一组包含两个元素的列表。
        # 其中，每一行通过指定的分隔符（split_）进行拆分，并且只保留包含两个元素的行。
        # 最后，以列表的形式返回两个元素的子列表，其中第一个元素是拆分后的第一个元素，第二个元素是拆分后的第二个元素，并去除首尾的空白字符。
        with open(file_name, "r", encoding="utf-8") as f:
            data = [i.split(split_) for i in f.readlines() if len(i.split(split_)) == 2]
            document_pair = [i[1].strip().split("-") for i in data]
        label_list = list()
        for i in document_pair:
            if len(i) == 2:
                label_list.append(i[1])
        all_label_set = list(set(label_list))
        all_label_set.sort(key=lambda x: label_list.index(x))
        categories_label2id = {j: i for i, j in enumerate(all_label_set, start=0)}
        return categories_label2id

    def collate_fn(self, batch):
        batch_token_ids, batch_labels = [], []
        for i, (text, text_labels) in enumerate(batch):
            tokens = self.tokenizer.tokenize(text, maxlen=self.max_len)
            mapping = self.tokenizer.rematch(text, tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            labels = np.zeros(
                (len(self.categories_label2id), self.max_len, self.max_len)
            )
            for start, end, label in text_labels:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = self.categories_label2id[label]
                    labels[label, start, end] = 1

            batch_token_ids.append(token_ids)  # 前面已经限制了长度
            batch_labels.append(labels[:, : len(token_ids), : len(token_ids)])
        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids), dtype=torch.long, device=self.device
        )
        batch_labels = torch.tensor(
            sequence_padding(batch_labels, seq_dims=3),
            dtype=torch.long,
            device=self.device,
        )
        return batch_token_ids, batch_labels

    def evaluate(self, data, threshold=0):
        """
        评估函数，计算F1、Precision、Recall
        :param data:
        :param threshold:
        :return:
        """
        X, Y, Z = 0, 1e-10, 1e-10
        for x_true, label in data:
            scores = self.model.predict(x_true)
            for i, score in enumerate(scores):
                R = set()
                for l, start, end in zip(*np.where(score.cpu() > threshold)):
                    R.add((start, end, self.categories_id2label[l]))

                T = set()
                for l, start, end in zip(*np.where(label[i].cpu() > 0)):
                    T.add((start, end, self.categories_id2label[l]))
                X += len(R & T)
                Y += len(R)
                Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def inference(self, text: str, threshold=0):
        """
        推理代码
        """
        tokens = self.tokenizer.encode(text, maxlen=self.max_len)[0]
        scores = self.model.predict(torch.tensor([tokens]))

        res = {}
        for l, start, end in zip(*np.where(scores[0].cpu() > threshold)):
            res[self.tokenizer.decode(tokens[start : end + 1])] = (
                self.categories_id2label[l]
            )
        return res

    def start_train(self, result_queue: Queue):
        """
        模型训练
        :param result_queue:
        :return:
        """
        try:
            evaluator = Evaluator(self)
            logger.info(
                f"命名实体识别模型开始训练，模型训练以及语料存储路径为：{self.model_dir}"
            )
            self.model.fit(
                self.train_dataloader,
                epochs=self.epochs,
                steps_per_epoch=None,
                callbacks=[evaluator],
            )
            logger.info("命名实体识别模型训练完成！！！")
            logger.info("*" * 20)
            logger.info(f"final_f1 -> {final_f1}")
            logger.info(f"final_precision -> {final_precision}")
            logger.info(f"final_recall -> {final_recall}")
            logger.info("*" * 20)
            result = {
                "sign": "train",
                "code": 200,
                "message": "命名实体识别模型训练成功",
                "data": {
                    "precision": final_precision,
                    "recall": final_recall,
                    "f1": final_f1,
                },
            }
            result_queue.put(result)
        except Exception as e:
            self.email_server.send_email_2_admin(
                "命名实体识别模型训练过程中出现异常",
                "【task_sequence_labeling_ner_global_pointer_bert4torch】训练过程中出现异常，停止训练！ -> {}".format(
                    e
                ),
            )
            logger.error(
                "命名实体识别模型训练过程中出现异常，停止训练！ -> {}".format(e)
            )
            # # 指定错误标志
            result = {
                "sign": "train",
                "code": 500,
                "message": "命名实体识别模型训练过程中出现异常，停止训练！ -> {}".format(
                    e
                ),
                "data": {},
            }
            result_queue.put(result)
        finally:
            pass

    def start_predict(self, predict_contents, result_queue: Queue):
        """
        模型预测
        :param predict_contents: 预测的内容列表
        :param result_queue: 结果队列，用于进程之间获取结果通信
        :return:
        """

        # 模型预测走CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            self.model.load_weights(self.model_dir + "/models/best_model.weights")

            results = []
            for content in predict_contents:
                # 这里是预测内容集合
                predict_result = self.inference(content)
                results.append(predict_result)
            result = {
                "sign": "predict",
                "code": 200,
                "message": "命名实体识别模型预测成功",
                "data": results,
            }
            logger.info("命名实体识别模型预测完成！！！")
            result_queue.put(result)
        except Exception as e:
            self.email_server.send_email_2_admin(
                "命名实体识别模型预测过程中出现异常",
                "【task_relation_extraction_gplinker_bert4torch】预测过程中出现异常！ -> {}".format(
                    e
                ),
            )
            logger.error("命名实体识别模型预测过程中出现异常！ -> {}".format(e))
            # # 指定错误标志
            result = {
                "sign": "predict",
                "code": 500,
                "message": "命名实体识别模型预测过程中出现异常！ -> {}".format(e),
                "data": {},
            }
            result_queue.put(result)


if __name__ == "__main__":
    train_dir = "D:/workspace_ai_train/xiaojingge-models/models/task_sequence_labeling_ner_global_pointer_bert4torch/corpus"
    queue = Queue()

    model_server = ModelServer(train_dir, "predict")
    # model_server.start_train(queue)
    model_server.start_predict(
        [
            "陈和生是个好人啊",
            "陈和生是个好人啊",
            "陈和生是个好人啊",
            "陈和生是个好人啊",
        ],
        queue,
    )
    print(queue.get())
