# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-11 9:09
@Author  : lijing
@File    : networks.py
@Description: 
---------------------------------------
'''

import os
import sys
import jieba
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from machine_learning.sentiment_analysis.sentiment_analysis_dict.utils import ToolGeneral
from machine_learning.sentiment_analysis.sentiment_analysis_dict.hyper_parameters import Hyperparams as hp

tool = ToolGeneral()
jieba.load_userdict(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict', 'jieba_sentiment.txt'))


class SentimentAnalysis():
    """
    进行情感分析
    """

    def sentiment_score_list(self, dataset):
        """
        计算情感得分列表

        参数:
        dataset (str): 需要进行情感分析的文本数据

        返回:
        list: 每个句子的情感得分列表，其中每个句子的得分列表包含积极和消极得分
        """
        # 使用工具方法分割句子
        seg_sentence = tool.sentence_split_regex(dataset)
        # 初始化句子的积极和消极得分列表
        results = []
        # 遍历每个句子
        for sentence in seg_sentence:
            # 使用结巴分词进行分词
            try:
                words = jieba.lcut(sentence, cut_all=False)
            except Exception as e:
                print(f"Error during sentence splitting: {e}")
                continue
            # 情感词位置
            a = 0
            scores = []
            # 遍历每个词
            for i, word in enumerate(words):
                # 初始化情感计数器
                pos_count, neg_count, pos_count2, neg_count2, pos_count3, neg_count3 = 0, 0, 0, 0, 0, 0
                # 如果词在积极词典中
                if word in hp.pos_dict:
                    # 处理特定的词组合
                    if word in ['好', '真', '实在'] and words[min(i + 1, len(words) - 1)] in hp.pos_neg_dict and words[min(i + 1, len(words) - 1)] != word:
                        continue
                    else:
                        # 增加积极词计数
                        pos_count += 1
                        # 情感词前的否定词数
                        c = 0
                        # 扫描情感词前的程度词
                        for w in words[a:i]:
                            # 根据程度词调整得分
                            if w in hp.most_dict:
                                pos_count *= 4
                            elif w in hp.very_dict:
                                pos_count *= 3
                            elif w in hp.more_dict:
                                pos_count *= 2
                            elif w in hp.ish_dict:
                                pos_count *= 0.5
                            elif w in hp.insufficiently_dict:
                                pos_count *= -0.3
                            elif w in hp.over_dict:
                                pos_count *= -0.5
                            elif w in hp.inverse_dict:
                                c += 1
                            else:
                                pos_count *= 1
                        # 扫描情感词前的否定词数, 根据否定词数调整得分
                        # 如果否定词数量为奇数，情感得分乘以-1.0（情感反转）
                        if tool.is_odd(c) == 'odd':
                            pos_count *= -1.0
                            pos_count2 += pos_count
                            pos_count3 += pos_count + pos_count2
                        else:
                            pos_count3 += pos_count
                        # 循环结束后统一重置 pos_count 和 pos_count2
                        pos_count = 0
                        pos_count2 = 0
                        a = i + 1
                # 消极情感的分析，与上面一致
                elif word in hp.neg_dict:
                    if word in ['好', '真', '实在'] and words[min(i + 1, len(words) - 1)] in hp.pos_neg_dict and words[min(i + 1, len(words) - 1)] != word:
                        continue
                    else:
                        neg_count += 1
                        # 情感词前的否定词数
                        d = 0
                        for w in words[a:i]:
                            if w in hp.most_dict:
                                neg_count *= 4
                            elif w in hp.very_dict:
                                neg_count *= 3
                            elif w in hp.more_dict:
                                neg_count *= 2
                            elif w in hp.ish_dict:
                                neg_count *= 0.5
                            elif w in hp.insufficiently_dict:
                                neg_count *= -0.3
                            elif w in hp.over_dict:
                                neg_count *= -0.5
                            elif w in hp.inverse_dict:
                                d += 1
                            else:
                                neg_count *= 1
                    # 如果否定词数量为奇数，情感得分乘以-1.0（情感反转）
                    if tool.is_odd(d) == 'odd':
                        neg_count *= -1.0
                        neg_count2 += neg_count
                        neg_count3 += neg_count + neg_count2
                    else:
                        neg_count3 += neg_count
                    # 循环结束后统一重置 neg_count 和 neg_count2
                    neg_count = 0
                    neg_count2 = 0
                    a = i + 1
                i += 1
                pos_count = pos_count3
                neg_count = neg_count3
                scores.append([pos_count, neg_count])
            # 处理感叹号
            # 扫描感叹号前的情感词，发现后权值*2
            # 感叹号前的情感得分翻倍
            if words and words[-1] in ['!', '！']:
                scores = [[j * 2 for j in c] for c in scores]

            # 处理转折词“但是”
            # 扫描但是后面的情感词，发现后权值*5
            for w_im in ['但是', '但']:
                if w_im in words:
                    ind = words.index(w_im)
                    ind = words.index(w_im)
                    scores_head = scores[:ind]
                    scores_tail = scores[ind:]
                    scores_tail_new = [[j * 5 for j in c] for c in scores_tail]
                    scores = scores_head + scores_tail_new
                    break
            # 处理问号，将得分设置为负面
            if words[-1] in ['?', '？']:  # 扫描是否有问好，发现后为负面
                scores = [[0, 2]]
            # 将句子得分添加到最终得分列表
            results.append(scores)
        # 返回最终的情感得分列表
        return results

    def sentiment_score(self, s):
        """
        计算输入文本的情感得分。

        该函数首先调用内部方法对输入文本进行情感分析，然后根据分析结果计算出正面和负面情感得分。

        参数:
        s (str): 需要进行情感分析的文本。

        返回:
        tuple: 包含两个浮点数，第一个是正面情感得分，第二个是负面情感得分。
        """
        # 调用内部方法获取情感得分列表
        senti_score_list = self.sentiment_score_list(s)

        # 初始化正面和负面情感得分列表
        positives = []
        negatives = []

        # 遍历情感得分列表，计算每个评论的正面和负面情感得分之和
        for review in senti_score_list:
            # 将评论的情感得分转换为NumPy数组，便于计算
            score_array = np.array(review)
            # 计算正面情感得分之和
            AvgPos = np.sum(score_array[:, 0])
            # 计算负面情感得分之和
            AvgNeg = np.sum(score_array[:, 1])
            # 将每个评论的正面和负面情感得分之和分别添加到对应的列表中
            negatives.append(AvgNeg)
            positives.append(AvgPos)

        # 计算所有评论的平均正面和负面情感得分
        pos_score = np.mean(positives)
        neg_score = np.mean(negatives)

        # 根据情感得分的正负调整得分值
        if pos_score >= 0 >= neg_score:
            # 如果正面得分非负而负面得分非正，则将负面得分取绝对值
            neg_score = abs(neg_score)
        elif pos_score >= 0 and neg_score >= 0:
            # 如果正面和负面得分都非负，则保持得分不变
            pass

        # 如果情感得分列表为空，将正面和负面情感得分都设为0
        if not senti_score_list:
            pos_score, neg_score = 0, 0

        # 返回正面和负面情感得分
        return pos_score, neg_score

    def normalization_score(self, sent):
        """
        对给定的句子进行情感分数归一化处理。

        这个方法首先通过 self.sentiment_score 方法获取句子的正面和负面情感分数。
        然后根据这些分数的大小，决定归一化的方式，以反映句子的情感倾向性。

        参数:
        - sent: 输入的句子，类型为字符串。

        返回值:
        - _score1: 归一化后的正面情感分数。
        - _score0: 归一化后的负面情感分数。
        """

        _score1 = 0
        _score0 = 0

        # 获取句子的正面和负面情感分数
        score1, score0 = self.sentiment_score(sent)

        # 当正面和负面分数都大于4时，根据分数大小进行归一化
        if score1 > 4 and score0 > 4:
            if score1 >= score0:
                # 正面分数大于等于负面分数，正面分数设为1，负面分数按比例归一化
                _score1 = 1
                _score0 = score0 / score1
            elif score1 < score0:
                # 负面分数大于正面分数，负面分数设为1，正面分数按比例归一化
                _score0 = 1
                _score1 = score1 / score0
        else:
            # 当正面或负面分数小于等于4时，分数按比例归一化到0到1之间
            if score1 >= 4:
                _score1 = 1
            elif score1 < 4:
                _score1 = score1 / 4
            if score0 >= 4:
                _score0 = 1
            elif score0 < 4:
                _score0 = score0 / 4

        # 返回归一化后的正面和负面情感分数
        return _score1, _score0


if __name__ == '__main__':
    sa = SentimentAnalysis()
    text = '我妈说明儿不让出去玩'
    print(sa.normalization_score(text))
