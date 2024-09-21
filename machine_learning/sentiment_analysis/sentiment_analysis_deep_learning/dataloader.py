# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-13 18:38
@Author  : lijing
@File    : dataloader.py
@Description: 数据加载
---------------------------------------
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        suffix = os.path.splitext(file_path)[1]
        if suffix == ".csv":
            self.data = pd.read_csv(file_path)
        elif suffix == ".xlsx":
            self.data = pd.read_excel(file_path)
        self.texts = self.data["content"].tolist()
        self.labels = self.data["label"].tolist()

        # 用于存储无效数据
        invalid_data = []

        # 迭代每个数据项
        for index, item in enumerate(self.texts):
            if not (isinstance(item, str)):
                invalid_data.append(item)
                print("Invalid data:", item)
                self.texts.pop(index)
                self.labels.pop(index)

        # 使用LabelEncoder将标签转换为整数
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # 打印每个类别的数量
        label_count = Counter(self.labels)
        label_map = dict(
            zip(
                self.label_encoder.transform(self.label_encoder.classes_),
                self.label_encoder.classes_,
            )
        )
        for label, count in label_count.items():
            # print(f"Label {label}: {count} samples")
            print(f"Label {label_map[label]} ({label}): {count} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item
