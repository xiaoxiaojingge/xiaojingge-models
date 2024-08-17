# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-08-14 16:08
@Author  : lijing
@File    : svm.py
@Description: 使用svm实现文本情感分类
---------------------------------------
'''
import jieba
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.pipeline import Pipeline

df_pos = pd.read_csv('dataset/pos.csv', encoding='utf-8', sep='\t')
df_pos = df_pos.dropna()

df_neg = pd.read_csv('dataset/neg.csv', encoding='utf-8', sep='\t')
df_neg = df_neg.dropna()

pos = df_pos.content.values.tolist()[1000:11000]
neg = df_neg.content.values.tolist()[1000:11000]

stopwords = pd.read_csv('./row_data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

print(len(stopwords))


def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception:
            continue


sentences = []
preprocess_text(pos, sentences, 'pos')
preprocess_text(neg, sentences, 'neg')

# 打印文本集合的长度
print(len(sentences))

random.shuffle(sentences)
# print(sentences[0:10])

x, y = zip(*sentences)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
# print(f'len(x_train): {len(x_train)}')
# print(f'len(x_test): {len(x_test)}')
# print(f'len(y_train): {len(y_train)}')
# print(f'len(y_test): {len(y_test)}')

# 使用TFIDF模型和SVM进行模型训练
tfidf_svm_sentiment_model = Pipeline([('TFIDF', TfidfVectorizer()), ('SVM', SVC(C=0.95, kernel="linear", probability=True))])
tfidf_svm_sentiment_model.fit(x_train[:10000], y_train[:10000])
svm_test_score = tfidf_svm_sentiment_model.score(x_test, y_test)
joblib.dump(tfidf_svm_sentiment_model, './products/tfidf_svm_sentiment.model')
print(svm_test_score)
print("svm模型训练完成")


# model = joblib.load('products/tfidf_svm_sentiment.model')
# 判断句子消极还是积极
def IsPoOrNeg(text):
    # 加载训练好的模型
    #     model = joblib.load('tfidf_nb_sentiment.model')

    # 去除停用词
    # text = remove_stropwords(text, cachedStopWords)
    # jieba分词
    seg_list = jieba.cut(text, cut_all=False)
    text = " ".join(seg_list)
    # 否定不处理
    # text = Jieba_Intensify(text)
    # y_pre = model.predict([text])
    # print(y_pre)
    proba = model.predict_probag([text])[0]
    # print(proba)
    if proba[1] > 0.4:
        print(text, ":此话极大可能是积极情绪（概率：）" + str(proba[1]))
        return "积极"
    else:
        print(text, ":此话极大可能是消极情绪（概率：）" + str(proba[0]))
        return "消极"

# IsPoOrNeg("好大的味道，放了三四天了那个味都去不了。垃圾货来的")
