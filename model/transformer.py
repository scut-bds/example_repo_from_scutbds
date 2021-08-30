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

# Model example
# Author: Chen Xiaofeng, Chen Yirong
# Date: 2021.01.05
# Reference: 
# [1] https://github.com/tensorflow/tensor2tensor
# [2] https://github.com/huggingface/transformers

import math
import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.max_len = config["n_positions"]
        self.embedding_size = config["n_embd"]
        self.word_embedding = nn.Embedding(config["vocab_size"], self.embedding_size, padding_idx=1)

    def forward(self, ids, token_type_ids=None):
        batch_size = ids.size(0)
        sequence_len = ids.size(1)
        position_seq = torch.arange(0, self.max_len).unsqueeze(1).repeat(1, int(self.embedding_size/2)).float()
        div = torch.arange(0, self.embedding_size, 2).repeat(self.max_len, 1).float()
        posi_matrix = position_seq/10000**(div/self.embedding_size)
        position_embedding = torch.cat([torch.sin(posi_matrix), torch.cos(posi_matrix)], dim=-1)
        position_vector = position_embedding[: sequence_len].repeat(batch_size, 1, 1).to(ids.device)
        word_vector = self.word_embedding(ids)
        if token_type_ids is None:
            token_type_vector = torch.zeros_like(word_vector, device=word_vector.device)
        else:
            token_type_vector = self.word_embedding(token_type_ids)
        padding_len = [torch.sum(ids[i] == 1).item() for i in range(batch_size)]
        return (word_vector+position_vector+token_type_vector), padding_len


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.ff_in = nn.Linear(config["n_embd"], 4*config["n_embd"])
        self.ff_out = nn.Linear(4*config["n_embd"], config["n_embd"])
        self.acti_func = gelu
        self.resid_dropout = nn.Dropout(config["resid_pdrop"])
        self.layer_norm = LayerNorm(config["n_embd"])

    def forward(self, x):
        return self.layer_norm(x + self.ff_out(self.resid_dropout(self.acti_func(self.ff_in(x)))))


class MultiHeadAttn(nn.Module):
    def __init__(self, config, qkv_size=None):
        super(MultiHeadAttn, self).__init__()
        self.num_head = config["n_head"]
        if qkv_size:
            self.qkv_size = qkv_size
        else:
            self.qkv_size = int(config["n_embd"] / config["n_head"])
        self.Q = nn.Linear(config["n_embd"], self.qkv_size * self.num_head)
        self.K = nn.Linear(config["n_embd"], self.qkv_size * self.num_head)
        self.V = nn.Linear(config["n_embd"], self.qkv_size * self.num_head)
        self.outputlinear = nn.Linear(self.qkv_size * self.num_head, config["n_embd"])
        self.attn_dropout = nn.Dropout(config["attn_pdrop"])
        self.layer_norm = LayerNorm(config["n_embd"])

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query_sl = query.size(1)
        key_sl = key.size(1)
        q, k, v = self.Q(query), self.K(key), self.V(value)
        q = q.view(batch_size, query_sl, self.num_head, self.qkv_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, key_sl, self.num_head, self.qkv_size).permute(0, 2, 3, 1)
        v = v.view(batch_size, key_sl, self.num_head, self.qkv_size).permute(0, 2, 1, 3)
        attn_score = torch.matmul(q, k) / self.qkv_size**0.5
        if mask is not None:
            mask = mask.repeat(1, self.num_head, 1)
            mask = mask.view(batch_size, self.num_head, query_sl, key_sl)
            attn_score.masked_fill_(mask, -float("inf"))
        attn_weight = self.attn_dropout(torch.softmax(attn_score, dim=-1))
        output = torch.matmul(attn_weight, v).permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, -1, self.num_head * self.qkv_size)
        output = self.outputlinear(output)
        return self.layer_norm(query + output)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.context_attn = MultiHeadAttn(config)
        self.encoder_ff = FeedForward(config)

    def forward(self, context_vector, con_mask):
        return self.context_attn(context_vector, context_vector, context_vector, con_mask)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.response_attn = MultiHeadAttn(config)
        self.decode_attn = MultiHeadAttn(config)
        self.decoder_ff = FeedForward(config)

    def forward(self, context_vector, response_vector, enc_mask, dec_mask):
        response_seq_len = response_vector.size(1)
        decoder_hidden = self.response_attn(response_vector, response_vector, response_vector, dec_mask)
        decoder_hidden = self.decode_attn(decoder_hidden, context_vector, context_vector, enc_mask[:, :1].repeat(1, response_seq_len, 1))
        return self.decoder_ff(decoder_hidden)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.context_encoder = nn.ModuleList([Encoder(config) for _ in range(config["n_layer"])])
        self.response_decoder = nn.ModuleList([Decoder(config) for _ in range(config["n_layer"])])
        self.pro_linear = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, context_ids, token_type_ids, response_ids):
        con_seq_len = context_ids.size(1)
        context_vector, con_padding_len = self.embedding(context_ids, token_type_ids)
        con_mask = [[[1]*(con_seq_len-con_padding_len[i])+[0]*con_padding_len[i]] for i in range(len(con_padding_len))]
        con_mask = torch.tensor(con_mask, device=context_ids.device).repeat(1, con_seq_len, 1) == 0
        for encoder in self.context_encoder:
            context_vector = encoder(context_vector, con_mask)
        resp_seq_len = response_ids.size(1)
        response_vector, resp_padding_len = self.embedding(response_ids)
        resp_mask = [[[1]*(resp_seq_len-resp_padding_len[i])+[0]*resp_padding_len[i]] for i in range(len(resp_padding_len))]
        resp_mask = torch.tril(torch.tensor(resp_mask, device=response_ids.device).repeat(1, resp_seq_len, 1)) == 0
        for decoder in self.response_decoder:
            response_vector = decoder(context_vector, response_vector, con_mask, resp_mask)
        return self.pro_linear(response_vector)

