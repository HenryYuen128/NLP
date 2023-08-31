#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam
from tqdm import tqdm
from configparser import ConfigParser
from torch.optim import Optimizer
from pckgs.util import eval_metrics
from dataloader import load_dataset
from collections import Counter
from pckgs.util import tools
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from pckgs.BertFamily import BertTextCls
from pckgs.BertFamily.ptuningv2 import BertPrefixForSequenceClassification
from transformers import TrainingArguments, set_seed
from transformers import Trainer
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import EarlyStoppingCallback
import transformers
import numpy as np
import copy
from loguru import logger
from pckgs.util.tools import cm2df
from pckgs.util.data_process import Dataset, TestDataset, get_one_hot_label
import time
import os
import sys

from torchmetrics import MetricCollection, Precision, Recall
sys.path.append("..")


# from transformers import logging
logger.remove()
logger.add(sys.stderr)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import logging
# from ray.tune import CLIReporter

# import sys
# from trainer_base import BaseTrainer
# from transformers import AutoTokenizer, BertTokenizer, AutoConfig
# from ray import tune


# from model.p_tuning import BertPromptForSequenceClassification
# from model.bert import BertForSequenceClassification

# import optunat

# logger.remove()
# from loguru import logger


# logger.info(date) #输出：'2019-06-15 18:58:07'
# cf = configparser.ConfigParser()
# cf.read('conf/train.config', encoding="utf-8")
torch.manual_seed(0)

configParser = ConfigParser()
configParser.read("train_params.config")

############# basic conf ################
# plm models
pretrain_model = eval(configParser.get("pretrained_models", "pretrain_model"))
print(type(pretrain_model))


# model conf
plm_name = configParser.get("model_conf", "plm_name")
model_type = configParser.get("model_conf", 'model_type')
max_length = configParser.getint("model_conf", 'max_length')
problem_type = configParser.get("model_conf", 'problem_type')

# ptuning config
strategy = configParser.get("ptuning_config", "strategy")
pre_seq_len = configParser.getint("ptuning_config", "pre_seq_len")
prefix_projection = configParser.get("ptuning_config", "prefix_projection")
freeze_bert = configParser.getboolean("ptuning_config", "freeze_bert")
is_freeze_bert = '_FreezeBert' if freeze_bert else ""

# train conf
task_name = configParser.get("train_conf", "task_name")
trainset_path = configParser.get("train_conf", "train_set_path")
devset_path = configParser.get("train_conf", "dev_set_path")
testset_path = configParser.get("train_conf", "test_set_path")
text_col = configParser.get("train_conf", "text_col")
gradient_checkpointing = configParser.getboolean(
    "train_conf", "gradient_checkpointing")
batch_size = configParser.getint("train_conf", "batch_size")
lr = configParser.getfloat("train_conf", "lr")
early_stopping_patience = configParser.getint(
    "train_conf", "early_stopping_patience")
epoch_num = configParser.getint("train_conf", "epoch_num")
cls_threshold = configParser.getfloat("train_conf", "cls_threshold")
hidden_dropout_prob = configParser.getfloat(
    "train_conf", "hidden_dropout_prob")
WeightedBCELoss = configParser.getboolean("train_conf", "WeightedBCELoss")
FocalLoss = configParser.getboolean("train_conf", "FocalLoss")

is_cate_feat = configParser.getboolean("train_conf", "is_cate_feat")
cate_feat_col = configParser.get("train_conf", "cate_feat_col")
cate_feat_names = configParser.get("train_conf", "cate_feat_names")
cate_feat_hidden_size = configParser.getint(
    "train_conf", "cate_feat_hidden_size")
cate_feat_numclass = len(configParser.get(
    "train_conf", "cate_feat_names").split('|'))
if is_cate_feat:
    model_type = 'bert_w_cate_feat'


loss_type = None
if WeightedBCELoss:
    loss_type = "Weighted_BCE"
elif FocalLoss:
    loss_type = "FocalLoss"
else:
    loss_type = "BCE"


# output conf

date = time.strftime("%Y-%m-%d_%H:%M:%S")
pooling_method = configParser.get("train_conf", "pooling")
output_dir = f"{plm_name}_{model_type}_{loss_type}_{pooling_method}_{strategy}_{date}" if model_type != "ptuning" else f"{plm_name}_{model_type}_{loss_type}_{pooling_method}{is_freeze_bert}_{strategy}_{date}"
output_dir = os.path.join(task_name, output_dir)

# set logger
# logger = logger.opt()

logger.add(
    os.path.join(output_dir, "mylog.txt"),
    rotation="100 MB")


# logger.add(os.path.join(output_dir, "mylog.log"),
#            rotation="10 MB")
# logger.add(
#     sys.stdout,
#     enqueue=True
# )
# logger.opt()


# read data
label_col = configParser.get("train_conf", "label_col")
label_list = configParser.get("train_conf", "label_list").split('|')
logger.info(f"label column: {label_col}")
logger.info(f"label list: {label_list}")

logger.info(f"loss type: {loss_type}")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


now = datetime.now()


# pretrain_model = {'macbert_base': r'hfl/chinese-macbert-base',
#                   'macbert_large': r'hfl/chinese-macbert-large',
#                   'roberta_large': r'hfl/chinese-roberta-wwm-ext-large',
#                   'albert_tiny': r'/expand/Henry/model_zoo/albert-base-chinese-cluecorpussmall',
#                   'mengzi_base': r'Langboat/mengzi-bert-base',
#                   'roberta_base': r'/expand/Henry/model_zoo/chinese-roberta-wwm-ext',
#                   'con': r'/data/henry.yuan/code/PyTorchLearning/kol_cls/mengzi_contrastive_model',
#                   'lbert': r'hfl/chinese-lert-base',
#                   'lbert_con':r'/data/henry.yuan/code/PyTorchLearning/kol_cls/lbert_contrastive_model',
#                   'electra': r'/expand/Henry/model_zoo/electra-hongkongese-base-discriminator/',
#                   'con_roberta': r'/expand/Henry/code/AIA_NLP/iSay-NPS/contrastive_learning/output/aia_model/135'
#                   }

model_dict = {
    'ptuning': BertPrefixForSequenceClassification,
    'bert': BertTextCls.BertVanillaTextClsForTransformers,
    'bert_w_cate_feat': BertTextCls.BertClsWithCateFeat
}

plm_path = pretrain_model[plm_name]


if is_cate_feat:
    from pckgs.util.CustomTrainer import BaseTrainerWithCateVar as BaseTrainer
    train_set = load_dataset(filepath=trainset_path, label_col=label_col, label_list=label_list,
                             cate_feat_col=cate_feat_col, cate_feat_names=cate_feat_names)
    valid_set = load_dataset(filepath=devset_path, label_col=label_col, label_list=label_list,
                             cate_feat_col=cate_feat_col, cate_feat_names=cate_feat_names)
    test_set = load_dataset(filepath=testset_path, label_col=label_col, label_list=label_list,
                            cate_feat_col=cate_feat_col, cate_feat_names=cate_feat_names)
else:
    from pckgs.util.CustomTrainer import BaseTrainer
    train_set = load_dataset(filepath=trainset_path,
                             label_col=label_col, label_list=label_list)
    valid_set = load_dataset(filepath=devset_path,
                             label_col=label_col, label_list=label_list)
    test_set = load_dataset(filepath=testset_path,
                            label_col=label_col, label_list=label_list)
shuffle(train_set)

##################### Get trainset label weight ########################
# count_dict = train_set[label_col].value_counts().to_dict()
# pos_weight = list()
# label_freq_sum = train_set[label_col].count()
# each_label_inverse_freq = [label_freq_sum / count_dict[_label] for _label in label_list]
# pos_weight = [_each_label_inverse_freq / sum(each_label_inverse_freq) for _each_label_inverse_freq in each_label_inverse_freq]


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=plm_path,
)


train_encodings = tokenizer(train_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length')
valid_encodings = tokenizer(valid_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length')
test_encodings = tokenizer(test_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length')


logger.info(f"{'='*20} Input Sample {'='*20}")
for i in range(5):
    logger.info(train_set[text_col].tolist()[i])
    logger.info(train_encodings['input_ids'][i])
    logger.info(train_set[label_col].tolist()[i])
    logger.info(train_set['one_hot_label'].tolist()[i])
    if is_cate_feat:
        logger.info(train_set['cate_feat_one_hot'].tolist()[i])
logger.info(f"{'='*20} Input Sample {'='*20}")

# DataLoader
if is_cate_feat:

    train_dataset = Dataset(train_set['one_hot_label'].tolist(),
                            train_encodings,
                            train_set[text_col].tolist(),
                            train_set['cate_feat_one_hot'].tolist())
    eval_dataset = Dataset(valid_set['one_hot_label'].tolist(),
                           valid_encodings,
                           valid_set[text_col].tolist(),
                           valid_set['cate_feat_one_hot'].tolist())
    test_dataset = Dataset(test_set['one_hot_label'].tolist(),
                           test_encodings,
                           test_set[text_col].tolist(),
                           test_set['cate_feat_one_hot'].tolist())
else:

    train_dataset = Dataset(train_set['one_hot_label'].tolist(),
                            train_encodings,
                            train_set[text_col].tolist())

    eval_dataset = Dataset(valid_set['one_hot_label'].tolist(),
                           valid_encodings,
                           valid_set[text_col].tolist())

    test_dataset = Dataset(test_set['one_hot_label'].tolist(),
                           test_encodings,
                           test_set[text_col].tolist())


label_mat = np.array(train_set['one_hot_label'].tolist())
logger.info(
    "=============================Trainset label info=============================")
label_cnt = np.sum(label_mat, axis=0)
logger.info(f"{label_cnt}")
logger.info(
    "=============================Trainset label info=============================")

# train_set['one_hot_label'].n

if WeightedBCELoss:

    total_label_cnt = np.sum(label_cnt)
    weights = (total_label_cnt / label_cnt).tolist()

else:
    weights = [1] * len(label_list)
logger.info(f"weights: {weights}")


logger.info('-' * 20)
logger.info('Param \n')
logger.info('Model Type: {}'.format(model_type))
logger.info('Prefix_projection Type: {}'.format(prefix_projection))
logger.info('pre_seq_len: {}'.format(pre_seq_len))
logger.info('Strategy: {}'.format(strategy))


config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=plm_path,

)

config.hidden_dropout_prob = hidden_dropout_prob
config.pre_seq_len = pre_seq_len
# For PrefixEncoder with 2-layer MLP
config.prefix_projection = prefix_projection.lower()
config.prefix_hidden_size = 512
config.strategy = strategy
config.output_hidden_states = True
config.num_labels = len(label_list)
config.problem_type = problem_type
config.use_cache = False
config.freeze_bert = freeze_bert
config.use_gradient_checkpointing = gradient_checkpointing
config.WeightedBCELoss = WeightedBCELoss
config.FocalLoss = FocalLoss
config.bce_pos_weight = weights
config.pooling = pooling_method
if cate_feat_hidden_size:
    config.cate_feat_hidden_size = cate_feat_hidden_size
    config.cate_feat_numclass = cate_feat_numclass

# train from scratch
model = model_dict[model_type.lower()].from_pretrained(
    plm_path,
    config=config
)

if is_cate_feat:
    logger.info(
        f"***************************** Train BERT with addition Categorical Feature *****************************")


def compute_metrics(pred, labels):
    from sklearn.metrics import precision_recall_fscore_support
    pred_array = np.concatenate(pred)
    labels_array = np.concatenate(labels)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=labels_array, y_pred=pred_array, average='macro')
    print(f"P:{p}, R:{r}, F1:{f1}")
    return p, r, f1

# continuous training
# model = BertPrefixForSequenceClassification.from_pretrained('/data/haobin/code/few-short-learning-research/slib/result/prefix/vanilla/7/False/output/checkpoint-2550',
    # config=config)


# model = BertPrefixForSequenceClassification.from_pretrained('/expand/Henry/code/AIA_NLP/iSay-NPS/multilabel_cls/roberta_base_ptuning_FocalLoss_weighted_avg_r_drop_2023-05-10_06:47:24/checkpoint-1008',
#                                                             config=config)
# from torch import optim
def train_eval(model,
               train_dataset,
               eval_dataset,
               batch_size,
               epochs,
               label_list,
               early_stopping_patience,
               output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_dataloader, eval_dataloader = DataLoader(dataset=train_dataset, num_workers=0, shuffle=True,
                                                   batch_size=batch_size), DataLoader(dataset=eval_dataset, num_workers=0,
                                                                                      shuffle=False, batch_size=batch_size)

    logger.info(len(train_dataloader))
    logger.info(len(eval_dataloader))

    logger.info('Device: {}'.format(device))

    loss_fct = nn.CrossEntropyLoss().cuda()

    optimizer = Adam([{'params': model.classifier.parameters(), 'lr': 1e-4},
                      {'params': model.bert.parameters()}],
                     lr=3e-5)

    optimizer = Adam(model.parameters(), lr=3e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-4, patience=5)
    model.to(device=device)
    # model = nn.DataParallel(model, device_ids=[2])
    # model.to(device)

    metric_collection = MetricCollection({
        'precision': MulticlassPrecision(num_classes=len(label_list), average='macro').to(device),
        'recall': MulticlassRecall(num_classes=len(label_list), average='macro').to(device),
        'f1': MulticlassF1Score(num_classes=len(label_list), average='macro').to(device)
    })

    logger.info(
        '------------------------------ Training ------------------------------')
    logger.info('Num Training Samples: {}'.format(len(train_dataset)))
    logger.info('Train Batch Size: {}'.format(batch_size))
    softmax = nn.Softmax(dim=-1)

    best_metrics = float('-inf')
    best_epoch = 0
    for epoch in tqdm(range(epochs), desc='Training and Evaluating...', position=0, leave=True):
        epoch += 1
        cur_epoch_train_loss = 0
        cur_epoch_eval_loss = 0
        model.train()

        for inputs in tqdm(train_dataloader, desc='Training...', position=0, leave=True):
            labels = inputs['labels'].to(device)

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            cate_feat = inputs['cate_feat'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, cate_feat=cate_feat)
            loss = loss_fct(logits, labels.float())

            cur_epoch_train_loss += loss.item()

            metric_collection.update(
                softmax(logits), torch.argmax(labels, dim=-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f'Training Info @ {epoch} epoch:')
        logger.info(f"Train loss: {cur_epoch_train_loss/len(train_dataset)}")

        logger.info(f"Train Metrics: {metric_collection.compute()}")
        metric_collection.reset()

        logger.info(
            "------------------------------ Evaluating ------------------------------")
        logger.info('Num Evaluation Samples: {}'.format(len(eval_dataset)))
        logger.info('Eval Batch Size: {}'.format(batch_size))

        with torch.no_grad():
            model.eval()
            pred_prob_list = list()
            true_label_list = list()
            for inputs in tqdm(eval_dataloader, desc='Evaluating...', position=0, leave=True):
                val_labels = inputs['labels'].to(device)

                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)
                cate_feat = inputs['cate_feat'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, cate_feat=cate_feat)

                val_loss = loss_fct(logits, val_labels.float())
                cur_epoch_eval_loss += val_loss.item()

                activation = torch.nn.Softmax()
                preds = activation(logits).cpu().detach().numpy()

                pred_prob_list.append(preds)
                true_label_list.append(val_labels.cpu().detach().numpy())

                pred_prbos_ndarray = np.concatenate(pred_prob_list)
                true_label_ndarray = np.concatenate(true_label_list)

                metric_collection.update(
                    softmax(logits), torch.argmax(val_labels, dim=-1))

        # compute_metrics(pred_list, label_list)
        logger.info(f'Evaluation Info @ {epoch} epoch:')
        logger.info(f"Eval loss: {cur_epoch_eval_loss/len(eval_dataset)}")

        metrics_dict = metric_collection.compute()
        logger.info(f'@ {epoch} Metrics: : {metrics_dict}')
        f1_score = metrics_dict.get('f1')
        if f1_score > best_metrics:
            best_metrics = f1_score
            best_epoch = epoch
        if epoch - best_epoch > early_stopping_patience:
            model.save_pretrained(output_dir)
            eval_metrics.compute_metrics_export_report(
                preds=pred_prbos_ndarray, y_true=true_label_ndarray, input_df=test_set, label_list=label_list, output_dir=output_dir)
            break
        logger.info(f"best f1: {best_metrics} @ epoch: {best_epoch}")

        metric_collection.reset()

        scheduler.step(val_loss)


train_eval(model,
           train_dataset=train_dataset,
           eval_dataset=eval_dataset,
           batch_size=batch_size,
           epochs=epoch_num,
           label_list=label_list,
           early_stopping_patience=early_stopping_patience,
           output_dir=output_dir
           )


#########  do predict ###################
model = model_dict[model_type.lower()].from_pretrained(
    r'/expand/Henry/code/AIA_NLP/iSay-NPS/level2_w_substage/roberta_base_bert_w_cate_feat_FocalLoss_mean_RDrop_2023-05-26_06:38:21',
    config=config
)
test_dataloader = DataLoader(dataset=test_dataset, num_workers=0, shuffle=False,
                             batch_size=batch_size)

with torch.no_grad():
    device = 'cuda'
    model.to(device)
    model.eval()
    pred_prob_list = list()
    true_label_list = list()
    for inputs in tqdm(test_dataloader, desc='Evaluating...', position=0, leave=True):
        val_labels = inputs['labels'].to(device)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        cate_feat = inputs['cate_feat'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids, cate_feat=cate_feat)

        activation = torch.nn.Softmax()
        preds = activation(logits).cpu().detach().numpy()

        pred_prob_list.append(preds)
        true_label_list.append(val_labels.cpu().detach().numpy())

    pred_prbos_ndarray = np.concatenate(pred_prob_list)
    true_label_ndarray = np.concatenate(true_label_list)

    eval_metrics.compute_metrics_export_report(preds=pred_prbos_ndarray, y_true=true_label_ndarray,
                                               input_df=test_set, label_list=label_list, output_dir=output_dir, prefix='test')
