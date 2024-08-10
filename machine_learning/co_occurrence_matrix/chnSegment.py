# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-10 13:55
@Author  : lijing
@File    : chnSegment.py
@Description: 
---------------------------------------
'''

from collections import Counter
from os import path
import jieba

jieba.load_userdict(path.join(path.dirname(__file__), 'userdict//userdict.txt'))  # 导入用户自定义词典

stopwords_path = 'userdict/stopword.txt'


def word_segment(text):
    '''
    通过jieba进行分词并通过空格分隔,返回分词后的结果
    '''

    # 移除一些无意义词语 'stopword.txt'
    my_word_list = []
    jieba.suggest_freq(('绿', '衣裳'), True)  # 人为设定需要拆分的词语
    seg_list = jieba.cut(text, cut_all=False)
    liststr = "/".join(seg_list)
    with open(stopwords_path, 'r', encoding='utf-8') as f_stop:
        f_stop_text = f_stop.read()
        f_stop_text = f_stop_text
    f_stop_seg_list = f_stop_text.split('\n')
    for myword in liststr.split('/'):  # 去除停顿词，生成新文档
        if not (myword.strip() in f_stop_seg_list) and len(myword.strip()) > 1:
            my_word_list.append(myword)
    text = ' '.join(my_word_list)

    # 词频存入txt文件
    dataDict = Counter(my_word_list)
    # 排序
    sorted_data = sorted(dataDict.items(), key=lambda item: item[1], reverse=True)
    with open('doc//词频统计.txt', 'w') as fw:
        for k, v in sorted_data:
            fw.write("%s,%d\n" % (k, v))
    print("词频统计:doc/词频统计.txt")
    return text
