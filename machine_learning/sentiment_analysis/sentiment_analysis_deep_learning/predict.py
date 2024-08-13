# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-13 19:29
@Author  : lijing
@File    : predict.py
@Description: 预测
---------------------------------------
'''
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from model import TextCNN
from model_config import Config
from tqdm import tqdm

# 配置
config = Config()

# 标签映射字典
label_map = {0: '书籍', 1: '平板', 2: '手机', 3: '水果', 4: '洗发水', 5: '热水器', 6: '蒙牛', 7: '衣服', 8: '计算机', 9: '酒店'}

# 加载tokenizer
try:
    print("尝试加载本地BertTokenizer...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    print("本地BertTokenizer加载成功！")
except Exception as e:
    print(f"无法加载本地tokenizer，请检查目录路径。错误信息：{e}")
    exit(0)


# 定义数据集类
class TestDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(file_path)
        self.texts = self.data['content'].tolist()
        self.ids = self.data['id'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['id'] = self.ids[idx]
        return item


def predict():
    # 加载测试数据
    test_dataset = TestDataset('./dataset/test.csv', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 初始化模型
    model = TextCNN(config.vocab_size, config.embed_size, config.num_classes, config.kernel_sizes, config.num_channels, config.dropout)
    device = torch.device(config.device)
    model.to(device)

    # 加载训练好的模型权重
    model_path = "./products/textcnn_model.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 预测
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            ids.extend(batch['id'])

    # 将预测结果写入test_with_predictions.csv
    test_data = pd.read_csv('./dataset/test.csv')
    test_data['label'] = [label_map[pred] for pred in predictions]
    test_data.to_csv('./dataset/test_with_predictions.csv', index=False)
    print("预测结果已保存到 ./dataset/test_with_predictions.csv")


if __name__ == '__main__':
    predict()
