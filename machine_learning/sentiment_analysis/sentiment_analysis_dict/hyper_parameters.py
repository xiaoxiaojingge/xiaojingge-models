# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-11 9:02
@Author  : lijing
@File    : hyper_parameters.py
@Description: 超参数信息，这里定义了词典
---------------------------------------
'''

import os
from machine_learning.sentiment_analysis.sentiment_analysis_dict.utils import ToolGeneral

pwd = os.path.dirname(os.path.abspath(__file__))
tool = ToolGeneral()

"""
deny_word：否定词词典，包含表示否定的词汇。
pos_dict：正面情感词典，包含表示正面情感的词汇。
neg_dict：负面情感词典，包含表示负面情感的词汇。
pos_neg_dict：合并的正面和负面情感词典。
most_dict：非常副词词典，包含表示非常程度的副词。
very_dict：很副词词典，包含表示很程度的副词。
more_dict：比较级副词词典，包含表示比较程度的副词。
ish_dict：程度不确定副词词典，包含表示程度不确定的副词。
insufficiently_dict：不足以程度副词词典，包含表示不足以程度的副词。
over_dict：过度程度副词词典，包含表示过度程度的副词。
inverse_dict：反转副词词典，包含表示反转的副词。
"""


class Hyperparams:
    '''Hyper parameters'''
    # Load sentiment dictionary
    deny_word = tool.load_dict(os.path.join(pwd, 'dict', 'not.txt'))
    pos_dict = tool.load_dict(os.path.join(pwd, 'dict', 'positive.txt'))
    neg_dict = tool.load_dict(os.path.join(pwd, 'dict', 'negative.txt'))
    pos_neg_dict = pos_dict | neg_dict
    # Load adverb dictionary
    most_dict = tool.load_dict(os.path.join(pwd, 'dict', 'most.txt'))
    very_dict = tool.load_dict(os.path.join(pwd, 'dict', 'very.txt'))
    more_dict = tool.load_dict(os.path.join(pwd, 'dict', 'more.txt'))
    ish_dict = tool.load_dict(os.path.join(pwd, 'dict', 'ish.txt'))
    insufficiently_dict = tool.load_dict(os.path.join(pwd, 'dict', 'insufficiently.txt'))
    over_dict = tool.load_dict(os.path.join(pwd, 'dict', 'over.txt'))
    inverse_dict = tool.load_dict(os.path.join(pwd, 'dict', 'inverse.txt'))
