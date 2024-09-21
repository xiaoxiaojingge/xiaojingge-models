# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-11 9:00
@Author  : lijing
@File    : utils.py
@Description: 工具类
---------------------------------------
"""

import re


class ToolGeneral:
    def is_odd(self, num):
        if num % 2 == 0:
            return "even"
        else:
            return "odd"

    def load_dict(self, file):
        """
        加载词典
        """
        with open(file, encoding="utf-8", errors="ignore") as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]
            print("Load data from file (%s) finished !" % file)
            dictionary = [word.strip() for word in lines]
        return set(dictionary)

    def sentence_split_regex(self, sentence):
        """
        将输入的句子sentence按照多种标点符号进行切分，并返回一个子句列表。
        替换破折号。
        按多种标点切分句子。
        移除空子串。
        返回子句列表或原句/空列表。
        """
        if sentence is not None:
            sentence = re.sub(r"&ndash;+|&mdash;+", "-", sentence)
            sub_sentence = re.split(
                r"[。,，！!？?;；\s…~～]+|\.{2,}|&hellip;+|&nbsp+|_n|_t", sentence
            )
            sub_sentence = [s for s in sub_sentence if s != ""]
            if sub_sentence != []:
                return sub_sentence
            else:
                return [sentence]
        return []


if __name__ == "__main__":
    #
    tool = ToolGeneral()
    #
    s = "我今天。昨天上午，还有现在"
    ls = tool.sentence_split_regex(s)
    print(ls)
