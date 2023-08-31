# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/13 16:08
@Author  : Henry.Yuan
@File    : tools.py

'''
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection == 'mlp':
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )



        elif self.prefix_projection == 'lstm':
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            # (b, pre_seq_len, hidden_size)
            # self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,
            #                                hidden_size=config.hidden_size // 2,
            #                                num_layers=2,
            #                                dropout=0.1,
            #                                bidirectional=True,
            #                                batch_first=True)
            self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,
                                           hidden_size=config.hidden_size // 2,
                                           num_layers=2,
                                           dropout=0.1,
                                           bidirectional=True,
                                           batch_first=True)

            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )

            # self.trans = torch.nn.Sequential(
            #     torch.nn.LSTM(config.hidden_size, config.prefix_hidden_size, bidirectional=True, batch_first=True),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            # )
        else:
            '''
            TODO:
            why config.num_hidden_layers * 2 * config.hidden_size ？
            '''
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection == 'mlp':
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        elif self.prefix_projection == 'lstm':
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(self.lstm_head(prefix_tokens)[0])
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values