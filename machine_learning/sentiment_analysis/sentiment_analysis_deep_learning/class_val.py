# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-13 19:07
@Author  : lijing
@File    : class_val.py
@Description: 
---------------------------------------
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model_config import Config
from model import TextCNN
from dataloader import TextDataset
from matplotlib.font_manager import FontProperties

config = Config()
# 验证集文件路径
val_file = "./dataset/valid.csv"
# 设置已训练好的模型权重
model_path = "./products/textcnn_model.pth"

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("./bert-base-chinese")

# 标签映射字典
label_map = {
    0: "书籍",
    1: "平板",
    2: "手机",
    3: "水果",
    4: "洗发水",
    5: "热水器",
    6: "蒙牛",
    7: "衣服",
    8: "计算机",
    9: "酒店",
}


def class_val():
    # 加载验证集数据集
    val_dataset = TextDataset(val_file, tokenizer)
    # 可根据需要调整 batch_size 和 num_workers
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    # 初始化模型
    model = TextCNN(
        config.vocab_size,
        config.embed_size,
        config.num_classes,
        config.kernel_sizes,
        config.num_channels,
        config.dropout,
    )
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    model.eval()

    # 验证模型
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["input_ids"], batch["labels"]
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算每个类别的准确率
    accuracy_per_class = {}

    for label_id, label_name in label_map.items():
        indices = [
            i for i, true_label in enumerate(true_labels) if true_label == label_id
        ]
        if len(indices) > 0:
            true_label_sublist = [true_labels[i] for i in indices]
            predicted_label_sublist = [predicted_labels[i] for i in indices]
            accuracy = accuracy_score(true_label_sublist, predicted_label_sublist)
            accuracy_per_class[label_name] = accuracy

    # 设置中文字体
    font = FontProperties(fname="./fonts/simhei.ttf")

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_per_class.keys(), accuracy_per_class.values())
    plt.xlabel("类别", fontproperties=font)
    plt.ylabel("准确率", fontproperties=font)
    plt.title("验证集每个类别的准确率", fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.tight_layout()
    plt.savefig(os.path.join("./products", "class_val_accuracy.png"))
    plt.show()


if __name__ == "__main__":
    class_val()
