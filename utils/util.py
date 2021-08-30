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

# File: util.py
# Used for dataset loading
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2021.08.30


import os
import re
import math
import json
import torch
import shutil
import collections
import pandas as pd
from os.path import join
from chardet import detect
from itertools import chain
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# reference: [CDial-GPT/od/](https://github.com/thu-coai/CDial-GPT)

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

#             input_ids: [CLS] [speaker1] 妈    [speaker2] 你 不 要 叫 我 妈 [speaker2] 我 不 是 你 妈 [SEP]
#        token_type_ids: [CLS] [DA1     ] Emo1] [DA2     ] [Emo2          ] [DA3     ] [Emo3       ] 
# positioning_embedding:   0       1       2         3      4 5  6  7  8  9      10    11 12 13 14 15 16


#             input_ids: [CLS] [speaker1] 妈         [speaker2] 你 不 要 叫 我 妈 [speaker2] 我 不 是 你 妈  [SEP]
#        token_type_ids: [CLS] [speaker1] [speaker1] [speaker2] [speaker1      ] [speaker2] [speaker2    ] [speaker2]
#     Emotion_embedding: [CLS] [Emo1               ] [Emo2                     ] [Emo3                              ]
#          DA_embedding: [CLS] [DA1                ] [DA2                      ] [DA3                               ]
# positioning_embedding:   0       1       2         3      4 5  6  7  8  9      10    11 12 13 14 15 16


# dialogue_text data_dir: /home/MMMTD/dialogue_text
# dialogue_video data_dir: /home/MMMTD/dialogue_video
# dialogue_audio data_dir: /home/MMMTD/dialogue_audio

DA_TOKENS = ["[greeting]","[question]","[answer]","[statement-opinion]","[statement-non-opinion]","[apology]",
             "[command]","[agreement]","[disagreement]","[acknowledge]","[appreciation]","[interjection]",
             "[conventional-closing]","[quotation]","[reject]","[irony]","[comfort]","[thanking]","[da-other]"]  # 19 DA labels

SENTIMENT_TOKENS = ["[neutral]","[positive]","[negative]"]

EMOTION_TOKENS = ["[happy]","[grateful]","[relaxed]","[positive-other]","[anger]","[sadness]","[fear]",
                  "[depress]","[disgust]","[astonished]","[worried]","[negative-other]","[neutral]"] # 13 emotion labels

BASEEMOTION_TOKENS = ["[happy]"]

DA_TO_TOKENS = {'greeting': '[greeting]', 'question': '[question]', 'answer': '[answer]', 
                'statement-opinion': '[statement-opinion]', 'statement-non-opinion': '[statement-non-opinion]', 
                'apology': '[apology]', 'command': '[command]', 'agreement': '[agreement]', 
                'disagreement': '[disagreement]', 'acknowledge': '[acknowledge]', 'appreciation': '[appreciation]', 
                'interjection': '[interjection]', 'conventional-closing': '[conventional-closing]', 
                'quotation': '[quotation]', 'reject': '[reject]', 'irony': '[irony]', 
                'comfort': '[comfort]','thanking':'[thanking]', 'other': '[da-other]'}

SENTIMENT_TO_TOKENS = {'neutral': '[neutral]', 'positive': '[positive]', 'negative': '[negative]'}

EMOTION_TO_TOKENS = {'happy': '[happy]', 'grateful': '[grateful]', 'relaxed': '[relaxed]', 
                     'positive-other': '[positive-other]', 'anger': '[anger]', 'sadness': '[sadness]', 
                     'fear': '[fear]', 'depress': '[depress]', 'disgust': '[disgust]', 
                     'astonished': '[astonished]', 'worried': '[worried]', 'negative-other': '[negative-other]', 
                     'neutral': '[neutral]'}

BASEEMOTION_TO_TOKENS = {"happy":'[happy]'}

AGEGROUP_TO_TOKENS = {"young":"young","middle-aged":"middle-aged","elderly":"elderly","teenager":"teenager","children":"children","unknown":"unknown"}


# for BERT ERC and DAC

DA_TO_ID = {'greeting': 0, 'question': 1, 'answer': 2, 'statement-opinion': 3, 'statement-non-opinion': 4, 
            'apology': 5, 'command': 6, 'agreement': 7, 'disagreement': 8, 'acknowledge': 9, 'appreciation': 10, 
            'interjection': 11, 'conventional-closing': 12, 'quotation': 13, 'reject': 14, 'irony': 15, 
            'comfort': 16,'thanking':17, 'other': 18}


EMOTION_TO_ID = {'happy': 0, 'grateful': 1, 'relaxed': 2, 'positive-other': 3, 'anger': 4, 'sadness': 5, 
                 'fear': 6, 'depress': 7, 'disgust': 8, 'astonished': 9, 'worried': 10, 
                 'negative-other': 11, 'neutral': 12}

GENDER_TO_ID = {'female': 0, 'unknown': 1, 'male': 2}
BIGFIVE_TO_ID = {'low': 0, 'unknown': 1, 'high': 2}



def get_data(args, tokenizer, data_path, logger):
    '''get_data
    Get .csv format dataset from data_path.
    
    '''
    logger.info("Read dataset from %s", data_path)
    data = pd.read_csv(data_path, 
                       usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"], 
                       encoding="UTF-8-SIG")
    # 有待增加一列base情感，利用"Emotion"列转换
    def createbaseemotion(emotion):
        new_emotion = emotion
        return new_emotion
    # data["BaseEmotion"] = [createbaseemotion(s) for s in data["Emotion"]]

    samples = data.iloc[0:30]
    logger.info("Start tokenizing and encoding the dataset")
    def tokenize(utterance):
        utterance = str(utterance)  # 保证为str类型
        # 对于问句添加问号
        utterance = utterance.replace("吗", "吗？")
        utterance = utterance.replace("？？", "？")

        # 对于感叹句添加感叹号
        utterance = utterance.replace("啊", "啊！")
        utterance = utterance.replace("吧", "吧！")
        utterance = utterance.replace("啦", "啦！")
        utterance = utterance.replace("呀", "呀！")
        utterance = utterance.replace("！！", "！")

        # 对于句子中间非问句，非感叹句添加逗号
        utterance = utterance.replace(" ", "，")
        # 去除重复标点符号
        utterance = utterance.split()  # 去除全部空格

        utt_list = list(utterance)  # "季杨杨，好像我听凡凡说过" --> ['季', '杨', '杨', '，', '好', '像', '我', '听', '凡', '凡', '说', '过']

        utterance = ' '.join(utt_list)  # ['季', '杨', '杨', '，', '好', '像', '我', '听', '凡', '凡', '说', '过']--> “季 杨 杨 ， 好 像 我 听 凡 凡 说 过”  # <class 'str'>
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utterance))
    data["Token"] = [tokenize(s) for s in data["Utterance"]]
    logger.info("Finished tokenizing and encoding the dataset")
    return data, samples


def convert_EMOTION_TO_TOKENS(emotion_list,emotion_type):
    emotion_tokens_list = []
    if emotion_type=="Sentiment":  # "Sentiment"
        for emo in emotion_list:
            if emo not in SENTIMENT_TO_TOKENS:
                emotion_tokens_list.append("[UNK]")
            else:
                emotion_tokens_list.append(SENTIMENT_TO_TOKENS[emo])
    elif emotion_type=="BaseEmotion":  # "BaseEmotion"
        for emo in emotion_list:
            if emo not in SENTIMENT_TO_TOKENS:
                emotion_tokens_list.append("[UNK]")
            else:
                emotion_tokens_list.append(BASEEMOTION_TO_TOKENS[emo])
    else:                                  # "Emotion"
        for emo in emotion_list:
            if emo not in SENTIMENT_TO_TOKENS:
                emotion_tokens_list.append("[UNK]")
            else:
                emotion_tokens_list.append(EMOTION_TO_TOKENS[emo])
    return emotion_tokens_list

def convert_DA_TO_TOKENS(da_list):
    da_tokens_list = []
    for da in da_list:
        da_tokens_list.append(DA_TO_TOKENS[da])
    return da_tokens_list


def create_speaker(speaker_list):
    speaker1 = speaker_list[0]
    new_speaker_list = []
    for speaker in speaker_list:
        if speaker==speaker1:
            new_speaker_list.append("[speaker1]")
        else:
            new_speaker_list.append("[speaker2]")
    return new_speaker_list


def set_da_in_speaker(da_ids,input_ids,bos, eos, pad, speaker1, speaker2):
    special_token_ids_list = [bos, eos, speaker1, speaker2]
    new_da_ids = []
    for i,da in enumerate(da_ids):
        if input_ids[i] in special_token_ids_list:
            new_da_ids.append(da_ids[i])
        else:
            new_da_ids.append(pad)
    return new_da_ids

def set_emotion_in_speaker(emotion_ids,input_ids,bos, eos, pad, speaker1, speaker2):
    special_token_ids_list = [bos, eos, speaker1, speaker2]
    new_emotion_ids = []
    for i,emotion in enumerate(emotion_ids):
        if input_ids[i] in special_token_ids_list:
            new_emotion_ids.append(emotion_ids[i])
        else:
            new_emotion_ids.append(pad)
    return new_emotion_ids





class MMMTDDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[greeting], [greeting], [greeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        token_type_ids: [ 0, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102]  # "[neutral]": 13102
    elif with_da==True:
        token_type_ids: [ 0, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088]  # "[greeting]": 13088
    else:
        token_type_ids: [ 0, 13086, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Sentiment", 
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type  # "Sentiment" or "BaseEmotion" or "Emotion"
        self.da_size = 18  # Number of DA categories
        self.emotion_size = 3  # Number of emotion categories
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        if self.emotion_type=="Sentiment":
            self.emotion_size = 3
        elif self.emotion_type=="BaseEmotion":
            self.emotion_size = 7
        else:  # self.emotion_type==2
            self.emotion_size = 15
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = data_index["Token"].tolist()[-1]
            

        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = []
        return self.process(speaker_list, utterance_history, emotion_list, da_list, response)



    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list

    def convert_EMOTION_TO_TOKENS(self,emotion_list):
        emotion_tokens_list = []
        if self.emotion_type=="Sentiment":  # "Sentiment"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[UNK]")
                else:
                    emotion_tokens_list.append(SENTIMENT_TO_TOKENS[emo])
        elif self.emotion_type=="BaseEmotion":  # "BaseEmotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[UNK]")
                else:
                    emotion_tokens_list.append(BASEEMOTION_TO_TOKENS[emo])
        else:                                  # "Emotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[UNK]")
                else:
                    emotion_tokens_list.append(EMOTION_TO_TOKENS[emo])
        return emotion_tokens_list

    def convert_DA_TO_TOKENS(self,da_list):
        da_tokens_list = []
        for da in da_list:
            da_tokens_list.append(DA_TO_TOKENS[da])
        return da_tokens_list


    def process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.with_da:
            instance["token_da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["token_emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.with_da:
            instance["token_da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["token_emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        if self.with_emotion:
            token_type_ids = pad_sequence(
                [torch.tensor(instance["token_emotion_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)

        elif self.with_da:
            token_type_ids = pad_sequence(
                [torch.tensor(instance["token_da_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)

        else:
            token_type_ids = pad_sequence(
                [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels



class EDADIALDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[greeting], [greeting], [greeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        emotion_ids:[ 0, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102]  # "[neutral]": 13102
    if with_da==True:
            da_ids: [ 0, 13103, 13103, 13103, 13103, 13103, 13103, 13103, 13103, 13103, 13103]  # "[greeting]": 13103
    token_type_ids: [ 0, 13103, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Sentiment", 
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type  # "Sentiment" or "BaseEmotion" or "Emotion"
        self.da_size = 18  # Number of DA categories
        self.emotion_size = 3  # Number of emotion categories
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        if self.emotion_type=="Sentiment":
            self.emotion_size = 3
        elif self.emotion_type=="BaseEmotion":
            self.emotion_size = 7
        else:  # self.emotion_type==2
            self.emotion_size = 15
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = data_index["Token"].tolist()[-1]
            

        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = []
        return self.process(speaker_list, utterance_history, emotion_list, da_list, response)



    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list

    def convert_EMOTION_TO_TOKENS(self,emotion_list):
        emotion_tokens_list = []
        if self.emotion_type=="Sentiment":  # "Sentiment"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(SENTIMENT_TO_TOKENS[emo])
        elif self.emotion_type=="BaseEmotion":  # "BaseEmotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(BASEEMOTION_TO_TOKENS[emo])
        else:                                  # "Emotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(EMOTION_TO_TOKENS[emo])
        return emotion_tokens_list

    def convert_DA_TO_TOKENS(self,da_list):
        da_tokens_list = []
        for da in da_list:
            da_tokens_list.append(DA_TO_TOKENS[da])
        return da_tokens_list


    def process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
        da_list = self.tokenizer.convert_tokens_to_ids(da_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
        da_list = self.tokenizer.convert_tokens_to_ids(da_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        if self.with_emotion:
            emotion_ids = pad_sequence(
                [torch.tensor(instance["emotion_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            emotion_ids = None

        if self.with_da:
            da_ids = pad_sequence(
                [torch.tensor(instance["da_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            da_ids = None

        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, emotion_ids, da_ids, labels


class  EDASDIALDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[greeting], [greeting], [greeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        emotion_ids:[ 0, 13102, -1, -1, 13102, -1, -1, 13102, -1, -1, -1]  # "[neutral]": 13102
    if with_da==True:
            da_ids: [ 0, 13103, -1, -1, 13103, -1, -1, 13103, -1, -1, -1]  # "[greeting]": 13103
    token_type_ids: [ 0, 13103, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Sentiment", 
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type  # "Sentiment" or "BaseEmotion" or "Emotion"
        self.da_size = 18  # Number of DA categories
        self.emotion_size = 3  # Number of emotion categories
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        if self.emotion_type=="Sentiment":
            self.emotion_size = 3
        elif self.emotion_type=="BaseEmotion":
            self.emotion_size = 7
        else:  # self.emotion_type==2
            self.emotion_size = 15
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = data_index["Token"].tolist()[-1]
            

        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
            emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
            da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            response = []
        return self.process(speaker_list, utterance_history, emotion_list, da_list, response)



    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list

    def convert_EMOTION_TO_TOKENS(self,emotion_list):
        emotion_tokens_list = []
        if self.emotion_type=="Sentiment":  # "Sentiment"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(SENTIMENT_TO_TOKENS[emo])
        elif self.emotion_type=="BaseEmotion":  # "BaseEmotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(BASEEMOTION_TO_TOKENS[emo])
        else:                                  # "Emotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(EMOTION_TO_TOKENS[emo])
        return emotion_tokens_list

    def convert_DA_TO_TOKENS(self,da_list):
        da_tokens_list = []
        for da in da_list:
            da_tokens_list.append(DA_TO_TOKENS[da])
        return da_tokens_list

    def set_da_in_speaker(self,da_ids,input_ids,bos, eos, speaker1, speaker2):
        special_token_ids_list = [bos, eos, speaker1, speaker2]
        new_da_ids = []
        for i,da in enumerate(da_ids):
            if input_ids[i] in special_token_ids_list:
                new_da_ids.append(da_ids[i])
            else:
                new_da_ids.append(self.pad)
        return new_da_ids

    def set_emotion_in_speaker(self,emotion_ids,input_ids,bos, eos, speaker1, speaker2):
        special_token_ids_list = [bos, eos, speaker1, speaker2]
        new_emotion_ids = []
        for i,emotion in enumerate(emotion_ids):
            if input_ids[i] in special_token_ids_list:
                new_emotion_ids.append(emotion_ids[i])
            else:
                new_emotion_ids.append(self.pad)
        return new_emotion_ids


    def process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
        da_list = self.tokenizer.convert_tokens_to_ids(da_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
            # only set the DA in [speaker1] or [speaker2]
            instance["da_ids"] = self.set_da_in_speaker(instance["da_ids"],instance["input_ids"],bos, eos, speaker1, speaker2)
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
            # only set the emotion in [speaker1] or [speaker2]
            instance["emotion_ids"] = self.set_emotion_in_speaker(instance["emotion_ids"],instance["input_ids"],bos, eos, speaker1, speaker2)
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, speaker_list, history, emotion_list, da_list, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
        da_list = self.tokenizer.convert_tokens_to_ids(da_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
            # only set the DA in [speaker1] or [speaker2]
            instance["da_ids"] = self.set_da_in_speaker(instance["da_ids"],instance["input_ids"],bos, eos, speaker1, speaker2)
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]
            # only set the emotion in [speaker1] or [speaker2]
            instance["emotion_ids"] = self.set_emotion_in_speaker(instance["emotion_ids"],instance["input_ids"],bos, eos, speaker1, speaker2)
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        if self.with_emotion:
            emotion_ids = pad_sequence(
                [torch.tensor(instance["emotion_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            emotion_ids = None

        if self.with_da:
            da_ids = pad_sequence(
                [torch.tensor(instance["da_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            da_ids = None

        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, emotion_ids, da_ids, labels



def build_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = MMMTDDataset(data=train_data, 
                                     tokenizer=tokenizer, 
                                     emotion_type=args.emotion_type, 
                                     max_history=args.max_history,
                                     batch_first=True, 
                                     lm_labels=True, 
                                     with_emotion=args.with_emotion, 
                                     with_da=args.with_da)

        valid_dataset = MMMTDDataset(data=valid_data, 
                                     tokenizer=tokenizer, 
                                     emotion_type=args.emotion_type, 
                                     max_history=args.max_history,
                                     batch_first=True, 
                                     lm_labels=True, 
                                     with_emotion=args.with_emotion, 
                                     with_da=args.with_da)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = MMMTDDataset(data=test_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type, 
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler


def build_edadial_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = EDADIALDataset(data=train_data, 
                                       tokenizer=tokenizer, 
                                       emotion_type=args.emotion_type, 
                                       max_history=args.max_history,
                                       batch_first=True, 
                                       lm_labels=True, 
                                       with_emotion=args.with_emotion, 
                                       with_da=args.with_da)

        valid_dataset = EDADIALDataset(data=valid_data, 
                                       tokenizer=tokenizer, 
                                       emotion_type=args.emotion_type, 
                                       max_history=args.max_history,
                                       batch_first=True, 
                                       lm_labels=True, 
                                       with_emotion=args.with_emotion, 
                                       with_da=args.with_da)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = EDADIALDataset(data=test_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type, 
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler


def build_edasdial_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = EDASDIALDataset(data=train_data, 
                                        tokenizer=tokenizer, 
                                        emotion_type=args.emotion_type, 
                                        max_history=args.max_history,
                                        batch_first=True, 
                                        lm_labels=True, 
                                        with_emotion=args.with_emotion, 
                                        with_da=args.with_da)

        valid_dataset = EDASDIALDataset(data=valid_data, 
                                        tokenizer=tokenizer, 
                                        emotion_type=args.emotion_type, 
                                        max_history=args.max_history,
                                        batch_first=True, 
                                        lm_labels=True, 
                                        with_emotion=args.with_emotion, 
                                        with_da=args.with_da)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = EDASDIALDataset(data=test_data, 
                                       tokenizer=tokenizer, 
                                       emotion_type=args.emotion_type, 
                                       max_history=args.max_history,
                                       batch_first=True, 
                                       lm_labels=True, 
                                       with_emotion=args.with_emotion, 
                                       with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler



class REDIALDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[greeting], [greeting], [greeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        token_type_ids: [ 0, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102]  # "[neutral]": 13102
    elif with_da==True:
        token_type_ids: [ 0, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088]  # "[greeting]": 13088
    else:
        token_type_ids: [ 0, 13086, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Sentiment", 
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type  # "Sentiment" or "BaseEmotion" or "Emotion"
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            current_speaker = speaker_list[-1] 
            current_emotion_id = EMOTION_TO_ID[data_index[self.emotion_type].tolist()[-1]]
            response = data_index["Token"].tolist()[-1]
            

        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            current_speaker = speaker_list[-1] 
            current_emotion_id = EMOTION_TO_ID[data_index[self.emotion_type].tolist()[-1]]
            response = []
        return self.process(speaker_list, utterance_history, current_speaker, current_emotion_id, response)


    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list


    def process(self, speaker_list, history, current_speaker, current_emotion_id, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        instance["current_emotion_id"] = current_emotion_id
        instance["lm_labels"] = [-1] * len(instance["input_ids"])


        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, speaker_list, history, current_speaker, current_emotion_id, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        instance["current_emotion_id"] = current_emotion_id
        instance["lm_labels"] = [-1] * len(instance["input_ids"])

        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        current_speaker_id = torch.tensor(
            [torch.tensor(instance["current_speaker_id"], dtype=torch.long) for instance in batch],
            dtype=torch.long)
        current_emotion_id = torch.tensor(
            [torch.tensor(instance["current_emotion_id"], dtype=torch.long) for instance in batch],
            dtype=torch.long)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, current_speaker_id, current_emotion_id, labels


def build_redial_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = REDIALDataset(data=train_data, 
                                      tokenizer=tokenizer, 
                                      emotion_type=args.emotion_type, 
                                      max_history=args.max_history,
                                      batch_first=True, 
                                      lm_labels=True, 
                                      with_emotion=args.with_emotion, 
                                      with_da=args.with_da)

        valid_dataset = REDIALDataset(data=valid_data, 
                                      tokenizer=tokenizer, 
                                      emotion_type=args.emotion_type, 
                                      max_history=args.max_history,
                                      batch_first=True, 
                                      lm_labels=True, 
                                      with_emotion=args.with_emotion, 
                                      with_da=args.with_da)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = REDIALDataset(data=test_data, 
                                     tokenizer=tokenizer, 
                                     emotion_type=args.emotion_type, 
                                     max_history=args.max_history,
                                     batch_first=True, 
                                     lm_labels=True, 
                                     with_emotion=args.with_emotion, 
                                     with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler




class RDADIALDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[gRDAeting], [gRDAeting], [gRDAeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        token_type_ids: [ 0, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102]  # "[neutral]": 13102
    elif with_da==True:
        token_type_ids: [ 0, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088]  # "[gRDAeting]": 13088
    else:
        token_type_ids: [ 0, 13086, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 da_type="DA", # 读取DA数据
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.da_type = da_type  # "DA" DA数据的列名
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            current_speaker = speaker_list[-1] 
            current_da_id = DA_TO_ID[data_index[self.da_type].tolist()[-1]]
            response = data_index["Token"].tolist()[-1]
            

        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            current_speaker = speaker_list[-1] 
            current_da_id = DA_TO_ID[data_index[self.da_type].tolist()[-1]]
            response = []
        return self.process(speaker_list, utterance_history, current_speaker, current_da_id, response)


    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list


    def process(self, speaker_list, history, current_speaker, current_da_id, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        instance["current_da_id"] = current_da_id
        instance["lm_labels"] = [-1] * len(instance["input_ids"])


        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, speaker_list, history, current_speaker, current_da_id, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        instance["current_da_id"] = current_da_id
        instance["lm_labels"] = [-1] * len(instance["input_ids"])

        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        current_speaker_id = torch.tensor(
            [torch.tensor(instance["current_speaker_id"], dtype=torch.long) for instance in batch],
            dtype=torch.long)
        current_da_id = torch.tensor(
            [torch.tensor(instance["current_da_id"], dtype=torch.long) for instance in batch],
            dtype=torch.long)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, current_speaker_id, current_da_id, labels




class CPEDDataset(Dataset):
    '''
    word_tokens:    [CLS] [speaker1] 您 好 [speaker2] 您 好 [speaker1] 再 见 [SEP]
    emotion_list:   [[neutral], [neutral], [neutral]]
    da_list:        [[gRDAeting], [gRDAeting], [gRDAeting]]
    input_ids：     [ 0, 13086, 448, 53, 13087, 448, 53, 13086, 154, 124, 2]
    if with_emotion==True:
        token_type_ids: [ 0, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102, 13102]  # "[neutral]": 13102
    elif with_da==True:
        token_type_ids: [ 0, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088, 13088]  # "[gRDAeting]": 13088
    else:
        token_type_ids: [ 0, 13086, 13086, 13086, 13087, 13087, 13087, 13086, 13086, 13086, 13086]
    labels：
    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Sentiment", # 读取DA数据
                 da_type="DA",
                 persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                 max_history=15, 
                 batch_first=True, 
                 lm_labels=True, 
                 with_current_speaker=False,
                 with_current_persona=False,
                 with_current_emotion=False,
                 with_current_da=False,
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type # 'Emotion' 情感标签列名
        self.da_type = da_type           # 'DA'      DA标签列名
        self.persona_type = persona_type
        self.with_current_speaker = with_current_speaker
        self.with_current_persona = with_current_persona
        self.with_current_emotion = with_current_emotion
        self.with_current_da = with_current_da
        self.with_emotion=with_emotion   # Whether use emotion to help generate dialogue
        self.with_da=with_da             # Whether use DA to help generate dialogue
        self.max_history = max_history   # Maximum number of dialogue sentences
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index]
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        if self.lm_labels:  # for train and valid dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            if self.with_emotion:
                emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
                emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            else:
                emotion_list = []
            if self.with_da:
                da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
                da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            else:
                da_list = []

            current_speaker = speaker_list[-1] 
            current_emotion_id = EMOTION_TO_ID[data_index[self.emotion_type].tolist()[-1]]
            current_da_id = DA_TO_ID[data_index[self.da_type].tolist()[-1]]
            if self.with_current_persona:
                current_gender_id = GENDER_TO_ID[data_index[self.persona_type[0]].tolist()[-1]]
                current_Neuroticism_id = BIGFIVE_TO_ID[data_index[self.persona_type[1]].tolist()[-1]]
                current_Extraversion_id = BIGFIVE_TO_ID[data_index[self.persona_type[2]].tolist()[-1]]
                current_Openness_id = BIGFIVE_TO_ID[data_index[self.persona_type[3]].tolist()[-1]]
                current_Agreeableness_id = BIGFIVE_TO_ID[data_index[self.persona_type[4]].tolist()[-1]]
                current_Conscientiousness_id = BIGFIVE_TO_ID[data_index[self.persona_type[5]].tolist()[-1]]
                current_persona_ids = [current_gender_id,current_Neuroticism_id,current_Extraversion_id,current_Openness_id,
                                       current_Agreeableness_id,current_Conscientiousness_id]
            else:
                current_persona_ids = []
            response = data_index["Token"].tolist()[-1]
            
        else:  # for test dataset
            speaker_list = self.create_speaker(data_index["Speaker"].tolist()[-2 * self.max_history:])
            utterance_history = data_index["Token"].tolist()[-2 * self.max_history:-1]
            if self.with_emotion:
                emotion_list = self.convert_EMOTION_TO_TOKENS(data_index[self.emotion_type].tolist()[-2 * self.max_history:])
                emotion_list = self.tokenizer.convert_tokens_to_ids(emotion_list)
            else:
                emotion_list = []
            if self.with_da:
                da_list = self.convert_DA_TO_TOKENS(data_index["DA"].tolist()[-2 * self.max_history:])
                da_list = self.tokenizer.convert_tokens_to_ids(da_list)
            else:
                da_list = []

            current_speaker = speaker_list[-1] 
            current_emotion_id = EMOTION_TO_ID[data_index[self.emotion_type].tolist()[-1]]
            current_da_id = DA_TO_ID[data_index[self.da_type].tolist()[-1]]
            if self.with_current_persona:
                current_gender_id = GENDER_TO_ID[data_index[self.persona_type[0]].tolist()[-1]]
                current_Neuroticism_id = BIGFIVE_TO_ID[data_index[self.persona_type[1]].tolist()[-1]]
                current_Extraversion_id = BIGFIVE_TO_ID[data_index[self.persona_type[2]].tolist()[-1]]
                current_Openness_id = BIGFIVE_TO_ID[data_index[self.persona_type[3]].tolist()[-1]]
                current_Agreeableness_id = BIGFIVE_TO_ID[data_index[self.persona_type[4]].tolist()[-1]]
                current_Conscientiousness_id = BIGFIVE_TO_ID[data_index[self.persona_type[5]].tolist()[-1]]
                current_persona_ids = [current_gender_id,current_Neuroticism_id,current_Extraversion_id,current_Openness_id,
                                       current_Agreeableness_id,current_Conscientiousness_id]
            else:
                current_persona_ids = []
            response = []
        return self.process(speaker_list, 
                            utterance_history,
                            emotion_list,
                            da_list, 
                            current_speaker, 
                            current_emotion_id,
                            current_da_id,
                            current_persona_ids, 
                            response)


    def create_speaker(self,speaker_list):
        speaker1 = speaker_list[0]
        new_speaker_list = []
        for speaker in speaker_list:
            if speaker==speaker1:
                new_speaker_list.append("[speaker1]")
            else:
                new_speaker_list.append("[speaker2]")
        return new_speaker_list


    def convert_EMOTION_TO_TOKENS(self,emotion_list):
        emotion_tokens_list = []
        if self.emotion_type=="Sentiment":  # "Sentiment"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(SENTIMENT_TO_TOKENS[emo])
        elif self.emotion_type=="BaseEmotion":  # "BaseEmotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(BASEEMOTION_TO_TOKENS[emo])
        else:                                  # "Emotion"
            for emo in emotion_list:
                if emo not in SENTIMENT_TO_TOKENS:
                    emotion_tokens_list.append("[neutral]")
                else:
                    emotion_tokens_list.append(EMOTION_TO_TOKENS[emo])
        return emotion_tokens_list

    def convert_DA_TO_TOKENS(self,da_list):
        da_tokens_list = []
        for da in da_list:
            da_tokens_list.append(DA_TO_TOKENS[da])
        return da_tokens_list

    def set_da_in_speaker(self,da_ids,input_ids,bos, eos, speaker1, speaker2):
        special_token_ids_list = [bos, eos, speaker1, speaker2]
        new_da_ids = []
        for i,da in enumerate(da_ids):
            if input_ids[i] in special_token_ids_list:
                new_da_ids.append(da_ids[i])
            else:
                new_da_ids.append(self.pad)
        return new_da_ids

    def set_emotion_in_speaker(self,emotion_ids,input_ids,bos, eos, speaker1, speaker2):
        special_token_ids_list = [bos, eos, speaker1, speaker2]
        new_emotion_ids = []
        for i,emotion in enumerate(emotion_ids):
            if input_ids[i] in special_token_ids_list:
                new_emotion_ids.append(emotion_ids[i])
            else:
                new_emotion_ids.append(self.pad)
        return new_emotion_ids


    def process(self, 
                speaker_list, 
                history, 
                emotion_list, 
                da_list, 
                current_speaker,
                current_emotion_id, 
                current_da_id, 
                current_persona_ids, 
                response, 
                with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        
        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]

        if self.with_current_speaker:
            instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        
        if self.with_current_emotion:
            instance["current_emotion_id"] = current_emotion_id
        
        if self.with_current_da:
            instance["current_da_id"] = current_da_id

        if self.with_current_persona:
            instance["current_persona_ids"] = current_persona_ids
        
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def testdata_process(self, 
                         speaker_list, 
                         history, 
                         emotion_list, 
                         da_list, 
                         current_speaker,
                         current_emotion_id, 
                         current_da_id, 
                         current_persona_ids, 
                         response, 
                         with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        speaker_list = self.tokenizer.convert_tokens_to_ids(speaker_list)
        instance = {}
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_list[i]] + s
                                    for i, s in enumerate(sequence[1:])]   
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_list[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]

        if self.with_da:
            instance["da_ids"] = [bos] + [da_list[i] for i, s in
                                                enumerate(sequence[1:])
                                                for _ in s]
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_list[i] for i, s in
                                                     enumerate(sequence[1:])
                                                     for _ in s]

        if self.with_current_speaker:
            instance["current_speaker_id"] = self.tokenizer.convert_tokens_to_ids(current_speaker)
        
        if self.with_current_emotion:
            instance["current_emotion_id"] = current_emotion_id
        
        if self.with_current_da:
            instance["current_da_id"] = current_da_id

        if self.with_current_persona:
            instance["current_persona_ids"] = current_persona_ids
        
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        if self.with_emotion:
            emotion_ids = pad_sequence(
                [torch.tensor(instance["emotion_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            emotion_ids = None

        if self.with_da:
            da_ids = pad_sequence(
                [torch.tensor(instance["da_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            da_ids = None

        if self.with_current_speaker:
            current_speaker_id = torch.tensor(
                [torch.tensor(instance["current_speaker_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_speaker_id = None
        
        if self.with_current_persona:
            current_persona_ids = pad_sequence(
                [torch.tensor(instance["current_persona_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=1) # padding_value=1 means unknown here
        else:
            current_persona_ids = None

        if self.with_current_emotion:
            current_emotion_id = torch.tensor(
                [torch.tensor(instance["current_emotion_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_emotion_id = None
        
        if self.with_current_da:
            current_da_id = torch.tensor(
                [torch.tensor(instance["current_da_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_da_id = None
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        

        return input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, labels


def build_cped_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = CPEDDataset(data=train_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type,
                                    da_type=args.da_type, 
                                    persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_current_speaker=args.with_current_speaker,
                                    with_current_persona=args.with_current_persona,
                                    with_current_emotion=args.with_current_emotion,
                                    with_current_da=args.with_current_da,
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)


        valid_dataset = CPEDDataset(data=valid_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type,
                                    da_type=args.da_type, 
                                    persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_current_speaker=args.with_current_speaker,
                                    with_current_persona=args.with_current_persona,
                                    with_current_emotion=args.with_current_emotion,
                                    with_current_da=args.with_current_da,
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = CPEDDataset(data=test_data, 
                                   tokenizer=tokenizer, 
                                   emotion_type=args.emotion_type,
                                   da_type=args.da_type, 
                                   persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                   max_history=args.max_history,
                                   batch_first=True, 
                                   lm_labels=True, 
                                   with_current_speaker=args.with_current_speaker,
                                   with_current_persona=args.with_current_persona,
                                   with_current_emotion=args.with_current_emotion,
                                   with_current_da=args.with_current_da,
                                   with_emotion=args.with_emotion, 
                                   with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler

















def build_rdadial_dataloaders(args, tokenizer, logger, load_test=False):
    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
        valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
        train_dataset = RDADIALDataset(data=train_data, 
                                      tokenizer=tokenizer, 
                                      da_type=args.da_type, 
                                      max_history=args.max_history,
                                      batch_first=True, 
                                      lm_labels=True, 
                                      with_emotion=args.with_emotion, 
                                      with_da=args.with_da)

        valid_dataset = RDADIALDataset(data=valid_data, 
                                      tokenizer=tokenizer, 
                                      da_type=args.da_type, 
                                      max_history=args.max_history,
                                      batch_first=True, 
                                      lm_labels=True, 
                                      with_emotion=args.with_emotion, 
                                      with_da=args.with_da)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data, test_samples = get_data(args, tokenizer, args.test_path, logger) # args.test_path="/home/MMMTD/dialogue_text/mmmtd_test_split.csv"
        test_dataset = RDADIALDataset(data=test_data, 
                                     tokenizer=tokenizer, 
                                     da_type=args.da_type, 
                                     max_history=args.max_history,
                                     batch_first=True, 
                                     lm_labels=True, 
                                     with_emotion=args.with_emotion, 
                                     with_da=args.with_da)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler





class BERTEDADataset(Dataset):
    ''' load data for train BertEDARC

    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 batch_first=True, 
                 with_emotion=False, 
                 with_da=False):
        self.data = data
        self.tokenizer = tokenizer
        self.with_emotion=with_emotion # Whether use emotion to help generate dialogue
        self.with_da=with_da # # Whether use DA to help generate dialogue
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.keys = list(set(self.data['Utterance_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        utterance_id = self.keys[index]
        data_index = self.data[self.data['Utterance_ID']==utterance_id]

        utterance = data_index["Token"].tolist()[0]
        emotion = EMOTION_TO_ID[data_index["Emotion"].tolist()[0]]
        da = DA_TO_ID[data_index["DA"].tolist()[0]]

        return self.process(utterance, emotion, da)


    def process(self, utterance, emotion, da):
        instance = {}
        instance["input_ids"] = utterance
        if self.with_emotion:
            instance["emotion_ids"] = emotion
        if self.with_da:
            instance["da_ids"] = da
        
        return instance


    def collate(self, batch):
        #input_ids = torch.tensor([torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch], dtype=torch.long)
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        if self.with_emotion:
            emotion_ids = torch.tensor([torch.tensor(instance["emotion_ids"], dtype=torch.long) for instance in batch],dtype=torch.long)
        else:
            emotion_ids = None

        if self.with_da:
            da_ids = torch.tensor([torch.tensor(instance["da_ids"], dtype=torch.long) for instance in batch],dtype=torch.long)
        else:
            da_ids = None

        return input_ids, emotion_ids, da_ids


def build_berteda_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")
    train_data,train_samples = get_data(args,tokenizer, args.train_path, logger) # args.train_path="/home/MMMTD/dialogue_text/mmmtd_train_split.csv"
    valid_data,valid_samples = get_data(args,tokenizer, args.valid_path, logger) # args.valid_path="/home/MMMTD/dialogue_text/mmmtd_valid_split.csv"
    train_dataset = BERTEDADataset(data=train_data, 
                                    tokenizer=tokenizer, 
                                    batch_first=False, 
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)

    valid_dataset = BERTEDADataset(data=valid_data, 
                                    tokenizer=tokenizer, 
                                    batch_first=False, 
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                collate_fn=train_dataset.collate,
                                num_workers=args.num_workers,
                                batch_size=args.train_batch_size,
                                shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, 
                                sampler=valid_sampler,
                                collate_fn=valid_dataset.collate,
                                num_workers=args.num_workers,
                                batch_size=args.valid_batch_size,
                                shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler

