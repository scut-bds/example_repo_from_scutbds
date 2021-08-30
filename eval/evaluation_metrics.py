# coding=utf-8
# Copyright 2021 South China University of Technology and 
# Engineering Research Ceter of Minstry of Education on Human Body Perception.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dialogue generation metrics file
# File: evaluation_metrics.py
# Used for evaluating the MMMTD dataset dialogue generation
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2021.08.30
# Reference: 
# [1] https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling


import re
import math
import nltk
import torch
from torch import nn
from torch import optim
from collections import defaultdict
from torch.nn import CrossEntropyLoss
import numpy as np
import bert_score  # version=='0.3.4'
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from transformers import BertModel, BertTokenizer


bert = BertModel.from_pretrained("bert-base-chinese")
token = BertTokenizer.from_pretrained("bert-base-chinese")


def read_file(file_path,num_lines=None):
    '''read corpus file in the format of .txt
       我 要 听 我 喝
       你 怎 么 知 道 这 能 解 酒 啊

       ---->
       list(list(str)), e.g.
       [['我', '要', '听', '我', '喝'],
        ['你', '怎', '么', '知', '道', '这', '能', '解', '酒', '啊']
       ]
    '''
    if num_lines==None:
        num_lines = len(open(file_path, encoding='utf-8').readlines())
    text_list = []
    for line in open(file_path,encoding='utf-8'):
        text_list.append(line.strip('\n').split())
        if len(text_list) == num_lines:
            break
    return text_list


def cal_average_text_len(text_list):
    '''calculate the average text lenth of the predicted corpus list
    Input: a list of text looks like:
       list(list(str))
       [['我', '要', '听', '我', '喝'],
        ['你', '怎', '么', '知', '道', '这', '能', '解', '酒', '啊']
       ]
    Output: float number

    '''
    text_num_list = []
    for text in text_list: # text is a 1-dim list, e.g. ['我', '要', '听', '我', '喝']
        text_num_list.append(len(text))
    average_lenth = np.mean(text_num_list)
    return average_lenth # >=1

def cal_diversity_1_2(text_list):
    '''calculate the diversity(D_1 and D_2)  of the predicted corpus list
    Input: a list of text looks like:
       list(list(str))
       [['我', '要', '听', '我', '喝'],
        ['你', '怎', '么', '知', '道', '这', '能', '解', '酒', '啊']
       ]
    Output: float list
       [D_1, D_2]

    '''
    tokens = [0.0,0.0]
    types = [defaultdict(int),defaultdict(int)]
    for text in text_list: # text is a 1-dim list, e.g. ['我', '要', '听', '我', '喝']
        for n in range(2):
            for idx in range(len(text)-n):
                ngram = ' '.join(text[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    D_1 = len(types[0].keys())/tokens[0]
    D_2 = len(types[1].keys())/tokens[1]
    return [D_1, D_2] # [0~1,0~1]

def cal_entropy(text_list):
    '''calculate the entropy of the predicted corpus list
    Input: a list of text looks like:
       list(list(str))
       [['我', '要', '听', '我', '喝'],
        ['你', '怎', '么', '知', '道', '这', '能', '解', '酒', '啊']
       ]
    Output: float list
       [0.0,0.0,0.0,0.0]       
    '''
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for text in text_list: # text is a 1-dim list, e.g. ['我', '要', '听', '我', '喝']
        for n in range(4):
            for idx in range(len(text)-n):
                ngram = ' '.join(text[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v /total * (np.log(v) - np.log(total))
    return etp_score # [0~1,0~1,0~1,0~1]


def cal_bleu(ref_list,pre_list):
    '''calculate the bleu score between the reference corpus list and the predicted corpus list
    Reference:
    http://www.javashuo.com/article/p-nzifylvm-gg.html
    http://www.voidcn.com/article/p-bcljwduq-btm.html
    Input:
    ref_list: list(list(str))
    pre_list: list(list(str))

    Output: float number

    '''
    bleu_1 = corpus_bleu(ref_list, pre_list, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(ref_list, pre_list, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(ref_list, pre_list, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(ref_list, pre_list, weights=(0, 0, 0, 1))
    # 如今的应用（paper）中评估BLEU值，通常取n-gram从1到4，并不作平均，而是作加和再取对数值。
    bleu_score = math.exp(bleu_1+bleu_2+bleu_3+bleu_4) # 0~e^4(54.598)
    return bleu_score

def greedy(x, x_words, y_words):
    '''
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    cosine = []  # 存放一个句子的一个词与另一个句子的所有词的余弦相似度
    sum_x = 0   # 存放最后得到的结果
    for x_v in x_words:
        for y_v in y_words:
            cosine.append(cosine_similarity(x_v, y_v))
        if cosine:
            sum_x += max(cosine)
            cosine = []
    x_len = len(x.split())#[:-1])
    sum_x = sum_x / x_len
    return sum_x

def cal_sentence_greedy_match(x, y):
    '''
    :param x: a sentence, here is a candidate answer
    :param y: a sentence, here is reference answer
    :return: a scalar in [0,1]
    '''
    if len(x) == 0:
        return 0

    x_words = word2vec(x)
    y_words = word2vec(y)
 
    # greedy match
    sum_x = greedy(x, x_words, y_words)
    sum_y = greedy(y, y_words, x_words)
    score = (sum_x+sum_y)/2
    return score

def cal_greedy_matching(ref_list,pre_list):
    '''calculate the Greedy Matching score between the reference corpus list and the predicted corpus list
    Input:
    ref_list: list(list(str))
    pre_list: list(list(str))
    Output:

    '''
    num_lines = len(pre_list)
    greedy_matching_score = 0.0
    for i in range(num_lines):
        prediction = pre_list[i] # ['我', '要', '听', '我', '喝']
        reference = ref_list[i]
        pre_utt = ' '.join(prediction) # "我 要 听 我 喝"
        ref_utt = ' '.join(reference)
        greedy_matching_score = greedy_matching_score + cal_sentence_greedy_match(pre_utt, ref_utt)
    return greedy_matching_score/num_lines # 0~1

def cal_bertscore(ref_list,pre_list):
    '''calculate the bertscore between the reference corpus list and the predicted corpus list
    Reference: https://github.com/Tiiiger/bert_score # version=='0.3.4'
    [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
    Input:
    pre_file: predictions.txt
    ref_file: references.txt
        each line in candicate_file and reference_file is like the follow:
        '你 好 好 学 习 ， 有 机 会 带 你 打 篮 球'
    '''
    cands = [' '.join(line) for line in pre_list]
    refs  = [' '.join(line) for line in ref_list]
    P, R, F1 = bert_score.score(cands, refs, lang='zh', verbose=True)
    bertscore = F1.mean().item()
    return bertscore # 0~1

def cal_sentence_embedding_average(x, y):
    #x_emb = word2senvec(x)
    #y_emb = word2senvec(y)
    if len(x) == 0:
        return 0
    x_vec = word2vec(x)
    y_vec = word2vec(y)

    x_emb = sentence_embedding(x_vec)
    y_emb = sentence_embedding(y_vec)

    embedding_average = EA_cosine_similarity(x_emb, y_emb)

    return embedding_average

def cal_embedding_average(ref_list,pre_list):
    '''calculate the embedding_average score between the reference corpus list and the predicted corpus list
    Input:
    ref_list: list(list(str))
    pre_list: list(list(str))
    Output:
    '''
    num_lines = len(pre_list)
    embedding_average_score = 0.0
    for i in range(num_lines):
        prediction = pre_list[i] # ['我', '要', '听', '我', '喝']
        reference = ref_list[i]
        pre_utt = ' '.join(prediction) # "我 要 听 我 喝"
        ref_utt = ' '.join(reference)
        embedding_average_score = embedding_average_score + cal_sentence_embedding_average(pre_utt, ref_utt)
    return embedding_average_score/num_lines # 0~1

def cal_sentence_bleu_score(reference, candidate, gram_id=1):
    '''cal_sentence_bleu_score
    reference: 参考答案, 形式：['季', '杨', '杨', '，', '好', '像', '我', '听', '凡', '凡', '说', '过']
    candidate: 模型预测答案,形式：同reference

    '''
    if gram_id == 1:
        if len(candidate) == 0:
            return 0
        return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    if gram_id == 2:
        if len(candidate) == 0:
            return 0
        return sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    if gram_id == 3:
        if len(candidate) == 0:
            return 0
        return sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    if gram_id == 4:
        if len(candidate) == 0:
            return 0
        return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return False

def cosine_similarity(x, y, norm=False):
    '''calculate the cosine_similarity between x and y
    '''
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
 
    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
 
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
 
 
def conver_float(x):
    float_str = x
    return [float(f) for f in float_str]

def word2vec(x):
    '''
    :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    input_ids = torch.tensor(token.encode(x)).unsqueeze(0)
    outputs = bert(input_ids)
    sequence_output = outputs[0]
    return sequence_output.tolist()[0]
 
 
def sentence_embedding(x_words):
    '''
    computing sentence embedding by computing average of all word embeddings of sentence.
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    sen_embed = np.array([0 for _ in range(len(x_words[0]))])  # 存放句向量
    #print(len(sen_embed))
 
    for x_v in x_words:
        x_v = np.array(x_v)
        #print(len(x_v))
        sen_embed = np.add(x_v, sen_embed)
    sen_embed = sen_embed / math.sqrt(sum(np.square(sen_embed)))
    return sen_embed


def word2senvec(x):
    '''
    将一个字符串(这里指句子）中所有的词都向量化，并存放到一个列表里
    :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    input_ids = torch.tensor(token.encode(x)).unsqueeze(0)
    outputs = bert(input_ids)
    sequence_output = outputs[1]
    return sequence_output.tolist()[0]


def EA_cosine_similarity(x, y, norm=False):
    '''(x) == len(y), "len(x) != len(y)"'''
    zero_list = np.array([0 for _ in range(len(x))])
    #print(zero_list)
    if x.all() == zero_list.all() or y.all() == zero_list.all():
        return float(1) if x == y else float(0)
 
    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
 
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def dialogue_bert_score(candicate_file, reference_file):
    ''' 
    Reference: https://github.com/Tiiiger/bert_score
    [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
    candicate_file: predictions.txt
    reference_file: references.txt
        each line in candicate_file and reference_file is like the follow:
        '你 好 好 学 习 ， 有 机 会 带 你 打 篮 球'

    '''
    with open(candicate_file) as f:
        cands = [line.strip() for line in f]

    with open(reference_file) as f:
        refs = [line.strip() for line in f]
    P, R, F1 = bert_score.score(cands, refs, lang='zh', verbose=True)
    bertscore = F1.mean().item()
    return bertscore


def read_test_result_for_eval(candidates_txt_file_path, references_txt_file_path):
    '''read_test_result_for_eval
    input, e.g.
        candidates_txt_file_path = "./generation_results/EDASDialGPT_with_emotion_False_with_da_True_predictions.txt"
        references_txt_file_path = "./generation_results/EDASDialGPT_with_emotion_False_with_da_True_references.txt"
    data format:
        你 好 好 学 习 ， 有 机 会 带 你 打 篮 球
        我 也 姓 杨

    '''
    with open(candidates_txt_file_path) as f:
        cands = [line.strip() for line in f]

    with open(references_txt_file_path) as f:
        refs = [line.strip() for line in f]
    return cands, refs

def cal_n_gram_score(pre_ids, tar_ids, N=4, mode="BLEU"):
    ''' Author: Chen Xiaofeng
    '''
    def build_n_gram_dict(ids, n):
        n_gram_dict = {}
        for i in range(1, n+1):
            n_gram_dict["%s-gram" % i] = {}
            for j in range(len(ids)-i+1):
                gram = tuple(ids[j: j+i])
                if gram in n_gram_dict["%s-gram" % i]:
                    n_gram_dict["%s-gram" % i][gram] += 1
                else:
                    n_gram_dict["%s-gram" % i][gram] = 1
        return n_gram_dict

    pre_n_gram_dict = build_n_gram_dict(pre_ids, N)

    if mode == "BLEU":
        tar_n_gram_dict = build_n_gram_dict(tar_ids, N)
        bp = math.exp(1 - len(tar_ids) / min(len(tar_ids), len(pre_ids)))
        score, num_n = 1, 0
        for n in pre_n_gram_dict:
            nume, demo = 0, 0
            if pre_n_gram_dict[n] != {}:
                num_n += 1
                for gram in pre_n_gram_dict[n]:
                    if gram in tar_n_gram_dict[n]:
                        nume += min(tar_n_gram_dict[n][gram], pre_n_gram_dict[n][gram])
                    demo += pre_n_gram_dict[n][gram]
                score *= nume / demo
            else:
                break
        n_gram_score = bp * (score ** (1 / num_n))
        return n_gram_score

    if mode == "Dist":
        n = "%s-gram" % N
        if pre_n_gram_dict[n] != {}:
            score = len(pre_n_gram_dict[n]) / len(pre_ids)
        else:
            score = 0
        return score

def get_losses_weights(losses:[list, np.ndarray, torch.Tensor]):
    '''多任务自适应损失权重
    reference:
    https://blog.csdn.net/leiduifan6944/article/details/107486857?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-2&spm=1001.2101.3001.4242
    '''
    if type(losses) != torch.Tensor:
        losses = torch.tensor(losses)
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights


class BertEDARC(nn.Module):
    '''
    the Finetune model for evaluating the dialogue generation model about 
    the emotion level and DA level
    '''
    def __init__(self, n_emo=13, n_da=19, pretrained_model="bert-base-chinese", task_name = "ERC"):
        '''
        n_emo: the number of emotion categories
        n_da: the number of DA categories
        task_mode: "ERC" or "DAC" or "EDAC"

        '''
        super(BertEDARC, self).__init__()
        model_class = BertModel
        #tokenizer_class = BertTokenizer
        self.n_emo = n_emo
        self.n_da = n_da
        self.task_name = task_name
        #self.tokenizer = tokenizer_class.from_pretrained(pretrained_model)
        self.bert = model_class.from_pretrained(pretrained_model)
        if self.task_name == "ERC":
            self.emo_ffn = nn.Linear(768, self.n_emo) # emotion classifier
        elif self.task_name == "DAC":
            self.da_ffn = nn.Linear(768, self.n_da) # DA classifier
        elif self.task_name == "EDAC":
            self.emo_ffn = nn.Linear(768, self.n_emo)
            self.da_ffn = nn.Linear(768, self.n_da)
        else:
            raise ValueError("task_mode need to be : \"ERC\" or \"DAC\" or \"EDAC\"")


    
    def forward(self, input_ids, emo_labels=None, da_labels=None):
        #for i in range(input_ids.size()[0])
        bert_output = self.bert(input_ids)
        bert_cls_hidden_state = bert_output[0][:,0,:]
        loss_fct = CrossEntropyLoss()
        if self.task_name == "ERC":
            outputs = self.emo_ffn(bert_cls_hidden_state)
            if emo_labels is not None:
                loss = loss_fct(outputs, emo_labels)
                outputs = (loss,) + outputs
            return outputs # (loss), emo_outputs
        if self.task_name == "DAC":
            outputs = self.da_ffn(bert_cls_hidden_state)
            if da_labels is not None:
                loss = loss_fct(outputs, da_labels)
                outputs = (loss,) + outputs
            return outputs # (loss), da_outputs
        if self.task_name == "EDAC":
            emo_outputs = self.emo_ffn(bert_cls_hidden_state)
            da_outputs = self.da_ffn(bert_cls_hidden_state)
            outputs = (emo_outputs,) + da_outputs

            if (emo_labels is not None) and (da_labels is not None):
                emo_loss = loss_fct(emo_outputs, emo_labels)
                da_loss = loss_fct(da_outputs, da_labels)
                old_loss = torch.cat([emo_loss,da_loss],dim=0)
                loss_w = get_losses_weights(old_loss)
                new_losses = old_loss * loss_w
                loss = torch.sum(new_losses)
                outputs = (loss,) + outputs
            return outputs  # (loss), emo_outputs, da_outputs
