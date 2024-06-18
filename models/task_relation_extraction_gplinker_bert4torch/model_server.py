# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2024-06-17 22:33
@Author  : lijing
@File    : model_server.py
@Description: 实体关系抽取模型（task_relation_extraction_gplinker_bert4torch）
三元组抽取任务，基于GlobalPointer的仿TPLinker设计
---------------------------------------
'''

# 配置相关
import configparser
# 系统相关
import os
# 路径相关
from pathlib import Path
# json相关
import json
# 日志相关
from config.logger import logger
# 邮件相关
from utils.email_util import EmailServer

# 模型相关
from bert4torch.layers import GlobalPointer
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset
from bert4torch.callbacks import Callback
from bert4torch.losses import SparseMultilabelCategoricalCrossentropy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# 多线程相关
from multiprocessing import Queue


class MyDataset(ListDataset):
    '''
    加载数据集
    '''

    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                D.append({'text': l['text'],
                          'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]})
        return D


class Model(BaseModel):
    '''
    定义bert上的模型结构
    '''

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.bert = build_transformer_model(params['bert_config_path'], params['bert_checkpoint_path'])
        self.entity_output = GlobalPointer(hidden_size=768, heads=2, head_size=64)
        self.head_output = GlobalPointer(hidden_size=768, heads=len(params['predicate2id']), head_size=64, RoPE=False,
                                         tril_mask=False)
        self.tail_output = GlobalPointer(hidden_size=768, heads=len(params['predicate2id']), head_size=64, RoPE=False,
                                         tril_mask=False)

    def forward(self, *inputs):
        hidden_states = self.bert(inputs)  # [btz, seq_len, hdsz]
        mask = inputs[0].gt(0).long()

        entity_output = self.entity_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        return entity_output, head_output, tail_output


class MyLoss(SparseMultilabelCategoricalCrossentropy):
    '''
    损失函数相关
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_trues):
        ''' y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]
        '''
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # [btz, heads, 实体起终点的下标]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))  # [btz, heads, seq_len*seq_len]
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return {'loss': sum(loss_list) / 3, 'entity_loss': loss_list[0], 'head_loss': loss_list[1],
                'tail_loss': loss_list[2]}


class SPO(dict):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo, params: dict):
        super().__init__()
        self.spox = (
            tuple(params['tokenizer'].tokenize(spo[0])), spo[1], tuple(params['tokenizer'].tokenize(spo[2]))
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self, model_server):
        self.best_val_f1 = 0.
        self.model_server = model_server

    def on_epoch_end(self, steps, epoch, logs=None):
        global final_f1, final_precision, final_recall
        f1, precision, recall = self.model_server.evaluate(self.model_server.valid_dataset.data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model_server.model.save_weights(self.model_server.model_dir + '/models/best_model_gplinker.pt')
            final_f1 = round(f1, 4)
            final_precision = round(precision, 4)
            final_recall = round(recall, 4)
        logger.info(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


class ModelServer:
    """
    模型训练类
    """

    def __init__(self, model_dir:str, sign: str):
        config = configparser.ConfigParser()
        # 读取.ini文件，这里的文件使用相对路径拼接
        config.read(str(Path(__file__).resolve().parent.parent.parent) + '\config\config.ini')
        # 获取bert模型相关信息
        bert_model_path = config.get('bert', 'model_path')

        self.model_dir = model_dir

        # 邮件服务
        self.email_server = EmailServer(
            config.get('email', 'smtp_server'),
            config.get('email', 'smtp_port'),
            config.get('email', 'sender_email'),
            config.get('email', 'sender_password'),
            config.get('email', 'receiver_email')
        )

        # 模型参数信息
        # 表示训练的最大长度。这个参数决定了输入序列的最大长度，超过这个长度的序列将被截断，而短于这个长度的序列将被填充。
        self.max_len = 128
        # 表示训练的批次大小。这个参数决定了每次训练的样本数量。
        self.batch_size = 32
        # 表示训练的轮数。这个参数决定了模型在训练数据上进行训练的次数。
        self.epochs = 1
        # 表示训练的学习率。这个参数决定了模型在训练过程中学习率如何调整，以控制模型的训练速度和精度。
        self.learning_rate = 1e-5

        # bert配置
        # 表示 BERT 模型的配置文件位置。该文件通常是一个 JSON 格式的文件，其中包含了 BERT 模型的各种配置参数，例如模型的层数、隐藏单元数、注意力头数等。配置文件提供了对模型结构进行修改和定制的方式。
        self.bert_config_path = bert_model_path + '/chinese_L-12_H-768_A-12/bert4torch_config.json'
        # 表示 BERT 模型的检查点位置。检查点文件通常是训练过程中保存的模型参数的二进制文件。通过加载这个文件，可以恢复训练过程中的模型参数，或者用于对未知数据进行预测。
        self.bert_checkpoint_path = bert_model_path + '/chinese_L-12_H-768_A-12/pytorch_model.bin'
        # 表示 BERT 模型的字典文件位置。字典文件是一个文本文件，其中包含了模型所使用的词汇表信息。BERT 模型将输入的文本转化为对应的词向量表示，字典文件定义了模型所使用的词汇和对应的编码方式。
        self.bert_dict_path = bert_model_path + '/chinese_L-12_H-768_A-12/vocab.txt'

        # 初始化其他实例变量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 建立分词器
        self.tokenizer = Tokenizer(self.bert_dict_path, do_lower_case=True)

        predicate2id, id2predicate = {}, {}

        with open(model_dir + '/all_schemas', encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                if l['predicate'] not in predicate2id:
                    id2predicate[len(predicate2id)] = l['predicate']
                    predicate2id[l['predicate']] = len(predicate2id)
        self.predicate2id = predicate2id
        self.id2predicate = id2predicate

        if sign == 'train':
            self.train_dataset = MyDataset(self.model_dir + '/train.json')
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                               collate_fn=self.collate_fn)
            self.valid_dataset = MyDataset(self.model_dir + '/valid.json')
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

        # 封装常用参数传递
        self.params = {
            'model_dir': self.model_dir,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'bert_config_path': self.bert_config_path,
            'bert_checkpoint_path': self.bert_checkpoint_path,
            'bert_dict_path': self.bert_dict_path,
            'predicate2id': self.predicate2id,
            'id2predicate': self.id2predicate,
            'evaluate': self.evaluate,
            'tokenizer': self.tokenizer
        }

        self.model = Model(self.params).to(self.device)
        self.model.compile(loss=MyLoss(mask_zero=True),
                           optimizer=optim.Adam(self.model.parameters(), self.learning_rate),
                           metrics=['entity_loss', 'head_loss', 'tail_loss'])

    def collate_fn(self, batch):
        def search(pattern, sequence):
            """从sequence中寻找子串pattern
            如果找到，返回第一个下标；否则返回-1。
            """
            n = len(pattern)
            for i in range(len(sequence)):
                if sequence[i:i + n] == pattern:
                    return i
            return -1

        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for d in batch:
            token_ids, segment_ids = self.tokenizer.encode(d['text'], maxlen=self.max_len)
            # 整理三元组 {s: [(o, p)]}
            spoes = set()
            for s, p, o in d['spo_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                sh = search(s, token_ids)
                oh = search(o, token_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
            # 构建标签
            entity_labels = [set() for _ in range(2)]
            head_labels = [set() for _ in range(len(self.predicate2id))]
            tail_labels = [set() for _ in range(len(self.predicate2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])  # [subject/object=2, 实体个数, 实体起终点]
            head_labels = sequence_padding(
                [list(l) for l in head_labels])  # [关系个数, 该关系下subject/object配对数, subject/object起点]
            tail_labels = sequence_padding(
                [list(l) for l in tail_labels])  # [关系个数, 该关系下subject/object配对数, subject/object终点]
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=self.device)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=self.device)
        # batch_entity_labels: [btz, subject/object=2, 实体个数, 实体起终点]
        # batch_head_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object起点]
        # batch_tail_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object终点]
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2), dtype=torch.float,
                                           device=self.device)
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2), dtype=torch.float,
                                         device=self.device)
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2), dtype=torch.float,
                                         device=self.device)
        return [batch_token_ids, batch_segment_ids], [batch_entity_labels, batch_head_labels, batch_tail_labels]

    def extract_spoes(self, text, threshold=0):
        """抽取输入text所包含的三元组
        """
        tokens = self.tokenizer.tokenize(text, maxlen=self.max_len)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        outputs = self.model.predict([token_ids, segment_ids])
        outputs = [o[0].cpu().numpy() for o in outputs]  # [heads, seq_len, seq_len]
        # 抽取subject和object
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= float('inf')
        outputs[0][:, :, [0, -1]] -= float('inf')
        for l, h, t in zip(*np.where(outputs[0] > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        # 识别对应的predicate
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[mapping[sh][0]:mapping[st][-1] + 1], self.id2predicate[p],
                        text[mapping[oh][0]:mapping[ot][-1] + 1]
                    ))
        return list(spoes)

    def evaluate(self, data):
        """评估函数，计算f1、precision、recall
        """
        global f1, precision, recall
        X, Y, Z = 0, 1e-10, 1e-10
        f = open(self.model_dir + '/dev_pred.json', 'w', encoding='utf-8')
        pbar = tqdm()
        for d in data:
            R = set([SPO(spo, self.params) for spo in self.extract_spoes(d['text'])])
            T = set([SPO(spo, self.params) for spo in d['spo_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
            s = json.dumps({'text': d['text'], 'spo_list': list(T), 'spo_list_pred': list(R),
                            'new': list(R - T), 'lack': list(T - R)}, ensure_ascii=False, indent=4)
            f.write(s + '\n')
        pbar.close()
        f.close()
        return f1, precision, recall

    def start_train(self, result_queue: Queue):
        '''
        模型训练
        :param result_queue:
        :return:
        '''
        try:
            evaluator = Evaluator(self)
            logger.info(f'实体关系抽取模型开始训练，模型训练以及语料存储路径为：{self.model_dir}')
            self.model.fit(model_server.train_dataloader, steps_per_epoch=None, epochs=self.epochs,
                           callbacks=[evaluator])
            logger.info('实体关系抽取模型训练完成！！！')
            logger.info('*' * 20)
            logger.info(f'final_f1 -> {final_f1}')
            logger.info(f'final_precision -> {final_precision}')
            logger.info(f'final_recall -> {final_recall}')
            logger.info('*' * 20)
            result = {
                "sign": 'train',
                "code": 200,
                "message": '实体关系抽取模型训练成功',
                "data": {
                    "precision": final_precision,
                    "recall": final_recall,
                    "f1": final_f1
                }
            }
            result_queue.put(result)
        except Exception as e:
            self.email_server.send_email_2_admin('实体关系抽取模型训练过程中出现异常',
                                                 '【task_relation_extraction_gplinker_bert4torch】训练过程中出现异常，停止训练！ -> {}'.format(
                                                     e))
            logger.error('实体关系抽取模型训练过程中出现异常，停止训练！ -> {}'.format(e))
            # # 指定错误标志
            result = {
                "sign": 'train',
                "code": 500,
                "message": '实体关系抽取模型训练过程中出现异常，停止训练！ -> {}'.format(e),
                "data": {}
            }
            result_queue.put(result)
        finally:
            pass

    def start_predict(self, predict_contents, result_queue: Queue):
        '''
        模型预测
        :param predict_contents: 预测的内容列表
        :param result_queue: 结果队列，用于进程之间获取结果通信
        :return:
        '''

        # 模型预测走CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            self.model.load_weights(self.model_dir + '/models/best_model_gplinker.pt')

            results = []
            for content in predict_contents:
                predict_result = self.extract_spoes(content)
                results.append(predict_result)
            result = {
                "sign": 'predict',
                "code": 200,
                "message": '实体关系抽取模型预测成功',
                "data": results
            }
            logger.info('实体关系抽取模型预测完成！！！')
            result_queue.put(result)
        except Exception as e:
            self.email_server.send_email_2_admin('实体关系抽取模型预测过程中出现异常',
                                                 '【task_relation_extraction_gplinker_bert4torch】预测过程中出现异常！ -> {}'.format(
                                                     e))
            logger.error('实体关系抽取模型预测过程中出现异常！ -> {}'.format(e))
            # # 指定错误标志
            result = {
                "sign": 'predict',
                "code": 500,
                "message": '实体关系抽取模型预测过程中出现异常！ -> {}'.format(e),
                "data": {}
            }
            result_queue.put(result)


if __name__ == '__main__':
    train_dir = 'D:/workspace_ai_train/re/35'
    queue = Queue()

    model_server = ModelServer(train_dir, 'train')
    model_server.start_train(queue)
    # model_server.start_predict(
    #     [
    #         '明确拍摄要求，提炼产品卖点，了解拍摄项目的内涵深入了解产品特征，挖掘拍摄项目亮点，通过画面有效表达',
    #         '设计拍摄思路与拍摄方法，创意各种风格和拍摄主题，对所有拍摄的内容及结果有准确预判，引导运营、设计师提升视觉效果',
    #         '产品拍摄，场景设计，与设计师配合，为后期制作提出优化调整方案',
    #         '1年以上摄影工作经验，热爱摄影，时尚触觉敏锐，能做到精益求精',
    #         '具备专业过硬的摄影技术，有创作基础，对色彩、构图、镜头语言有清晰认识',
    #         '具备较强的时尚感和色彩感，对摄影布光、道具摆设造型有独特的创意，擅长营造良好的产品氛围',
    #         '有较强的美术功底, 对色彩感觉强烈，视觉表达方面有个人独特观点',
    #         '具备商业摄影 / 电商摄影 / 广告公司 / 工作室行业经验',
    #         '这是一段测试文本，本岗位要求学历为本科，最好是物流管理专业，拥有新媒体运营职业技能等级证书(初级)证书，最好，会使用Office办公软件和数据分析工具Google Analytics，会使用uniapp,拥有英文四级证书最佳',
    #
    #     ],
    #     queue)
    print(queue.get())

