# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/8 16:05
@Author  : Henry.Yuan
@File    : data_process.py

'''


import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import IterableDataset
import transformers
import pandas as pd


def get_one_hot_label(cur_labels, label_list):
    return [1 if l in cur_labels.split('|') else 0 for l in label_list]


def load_dataset(filepath, label_col, label_list, cate_feat_col=None, cate_feat_names=None, is_test=False):
    df = pd.read_csv(filepath)
    
    print(f"dataset info:\n {df.info()}")
    # df = shuffle(df)
    if not is_test:
        # df[label_col] = df[label_col].fillna('NA')
        df['one_hot_label'] = df[label_col].apply(lambda x: get_one_hot_label(x, label_list))
    if cate_feat_col:
        df['cate_feat_one_hot'] = df[cate_feat_col].apply(lambda x: get_one_hot_label(x, cate_feat_names.split('|')))
    return df

def preprocess_dataset(df, label_col, label_list, cate_feat_col=None, cate_feat_names=None, is_test=False):
    print(f"dataset info:\n {df.info()}")
    if not is_test:
        # df[label_col] = df[label_col].fillna('NA')
        df['one_hot_label'] = df[label_col].apply(lambda x: get_one_hot_label(x, label_list))
    if cate_feat_col:
        df['cate_feat_one_hot'] = df[cate_feat_col].apply(lambda x: get_one_hot_label(x, cate_feat_names.split('|')))
    return df

class AdvancedTextClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset builder for text classification task

    Args:
    data_path: path to load data
    text_col: col includes Text
    label col: col includes label, single or multiple are both valid
    label list: list of target labels
    tokenizer: Transformers tokenizer
    
    """
    def __init__(self, data: pd.DataFrame, text_col: str, label_col: str, label_list: list, tokenizer: transformers.AutoTokenizer, max_length=128, is_test=False):
        self.df = preprocess_dataset(data, label_col=label_col, label_list=label_list, is_test=is_test)
        print(self.df.columns)
        # self.label_col = label_col
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.text_encodings = tokenizer(self.df[text_col].tolist(), max_length=max_length, truncation=True, padding='max_length', return_token_type_ids=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.text_encodings.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.df['one_hot_label'].tolist()[idx], dtype=float)
        # item['cate_feat'] = torch.tensor(self.substage_feat[idx], dtype=torch.int32) if self.substage_feat else None
        item['text'] = self.df[self.text_col].tolist()[idx]
        return item




# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, texts_encodings, texts, substage_feat=None, is_test=False):
        
        self.substage_feat = substage_feat
        self.texts_encodings = texts_encodings
        self.texts = texts
        self.is_test = is_test
        if not is_test:
            self.labels = labels

    def __len__(self):
        return len(self.texts)
    # def __getitem__(self, item):
    #     return self.texts_encodings[item], self.labels[item]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts_encodings.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=float)
        item['cate_feat'] = torch.tensor(self.substage_feat[idx], dtype=torch.int32) if self.substage_feat else None
        item['texts'] = self.texts[idx]
        return item


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, labels, texts_encodings, texts):
        self.labels = labels
        self.texts_encodings = texts_encodings
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    # def __getitem__(self, item):
    #     return self.texts_encodings[item], self.labels[item]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=float)
        item['texts'] = self.texts[idx]
        return item


def read_data(df, content_col, label_col, label_list, sep='|', with_label=True):
    texts = list()
    labels = list()
    df.dropna(subset=[content_col], inplace=True)
    for idx, row in tqdm(df.iterrows()):
        texts.append(row[content_col])

        if with_label:
            cur_label = row[label_col].split('|')
            labels.append([1 if l in cur_label else 0 for l in label_list])
        else:
            labels.append([1] * len(label_list))
    return texts, labels



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, texts):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=float)
        item['texts'] = self.texts[idx]
        return item

    def __len__(self):
        return len(self.labels)


def generate_attention_mask(token_mask, input_mask):
    token_mask[np.where(input_mask==0)] = -1
    token_mask = np.expand_dims(token_mask + 1, 0)
    # t_token_t = np.transpose(token_t)
    output_mask = (((token_mask ==np.transpose(token_mask)) & ((token_mask==1) | (token_mask==2)))).astype(int)
    output_mask = np.transpose(output_mask, [1, 0, 2])
    return output_mask


def custom_tokenizer(df, tokenizer, max_length=256):
    encodings = {'input_ids': list(), 'token_type_ids': list(), 'attention_mask': list()}
    for _, row in tqdm(df.iterrows()):
        # sent = row[col]

        sent = '[CLS]' + row['user_info'][:200] + '[SEP][CLS]' + row['kws'][:45] + '[SEP]'
        input_ids = tokenizer(sent, max_length=max_length, padding='max_length', add_special_tokens=False, return_tensors='np',
                              truncation=True)['input_ids'][0]
        cls_idx = np.where(input_ids == 101)
        sep_idx = np.where(input_ids == 102)
        assert sep_idx[0].shape[0] == 2


        # token_type_ids变为000,111,000
        token_type_ids = np.zeros_like(input_ids)
        token_type_ids[cls_idx[0][1]:sep_idx[0][1] + 1] = 1

        # attention_mask先变为000,111,222
        attention_mask = np.zeros_like(input_ids)
        attention_mask[:cls_idx[0][1]] = 1
        attention_mask[cls_idx[0][1]:sep_idx[0][1] + 1] = 2
        attention_mask = np.expand_dims(attention_mask, 0)

        # 提高改造好attention矩阵, 形状为seq len*seq len. 第一个CLS只注意到long text，第二个CLS只注意到key words
        output_attention_mask = (
        ((attention_mask == np.transpose(attention_mask)) & ((attention_mask == 1) | (attention_mask == 2)))).astype(int)

        # output_attention_mask[:cls_idx[0][1],:sep_idx[0][1]+1] = 1

        encodings['input_ids'].append(input_ids)
        encodings['token_type_ids'].append(token_type_ids)
        encodings['attention_mask'].append(output_attention_mask)
        # encodings['sentence'].append(sent)

    return encodings