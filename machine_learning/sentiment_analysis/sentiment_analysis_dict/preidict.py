# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-11 10:05
@Author  : lijing
@File    : preidict.py
@Description: 预测
---------------------------------------
'''

from machine_learning.sentiment_analysis.sentiment_analysis_dict.networks import SentimentAnalysis

SA = SentimentAnalysis()


def predict(sent):
    """
    根据句子的情感倾向进行预测。

    该函数通过比较句子的情感得分，来判断其情感倾向是积极、消极还是中性。
    它首先对输入的句子进行归一化处理，然后根据处理后的得分判断情感倾向。

    参数:
    sent (str): 输入的句子。

    返回值:
    int: 预测结果，1表示积极，-1表示消极，0表示中性。
    """
    # 初始化预测结果为0，代表中性
    result = 0
    # 使用SA模块的normalization_score函数对句子进行归一化处理并获取得分
    score1, score0 = SA.normalization_score(sent)
    print(score1, score0)
    # 如果积极和消极得分相等，判断为中性
    if score1 == score0:
        result = 0
    # 如果积极得分大于消极得分，判断为积极
    elif score1 > score0:
        result = 1
    # 如果积极得分小于消极得分，判断为消极
    elif score1 < score0:
        result = -1
    # 返回预测结果
    return result


if __name__ == '__main__':
    sentences = ['对你不满意', '对你挺满意的，因为你是个大美女', '帅哥', '我妈说明儿不让出去玩']
    for text in sentences:
        print(predict(text))
