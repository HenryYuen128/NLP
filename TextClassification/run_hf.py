#!/usr/bin/env python
# coding: utf-8

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.DataProcess import load_dataset, Dataset
from utils.CustomerTrainer import BaseTrainer
from configparser import ConfigParser
from utils import Metrics
from collections import Counter
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
import torch.nn
from models.TextClassification import BertFamily
from transformers import TrainingArguments, set_seed
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
import pandas as pd
from transformers import EarlyStoppingCallback
import numpy as np
from loguru import logger
import time

torch.manual_seed(3407)

# from transformers import logging
logger.remove()
logger.add(sys.stdout,
           format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
           "{process.name} | "  # 进程名
           "{thread.name} | "  # 进程名
           # 模块名.方法名
           "<cyan>{module}</cyan>.<cyan>{function}</cyan>"
           ":<cyan>{line}</cyan> | "  # 行号
           "<level>{level}</level>: "  # 等级
           "<level>{message}</level>",  # 日志内容
           )




configParser = ConfigParser()
configParser.read("/home/henry/code/NLP/TextClassification/params.ini")

############# basic conf ################

# model conf
plm_name = configParser.get("model_conf", "plm_name")
model_type = configParser.get("model_conf", 'model_type')
max_length = configParser.getint("model_conf", 'max_length')


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
problem_type = configParser.get("train_conf", 'problem_type')

is_cate_feat = configParser.getboolean("train_conf", "is_cate_feat")
cate_feat_col = configParser.get("train_conf", "cate_feat_col")
cate_feat_names = configParser.get("train_conf", "cate_feat_names")
add_vocab = configParser.getboolean("train_conf", "add_token")
vocab_path = configParser.get("train_conf", "vocab_path")


# output conf
pred_col_name = configParser.get(
    "output_config", "pred_output_col_name") if "pred_output_col_name" in configParser.options("train_conf") else "pred"
test_with_label = configParser.getboolean("output_config", "test_with_label")

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
output_dir = f"{plm_name.replace('/', '-')}_{model_type}_{loss_type}_{pooling_method}_{strategy}_{date}" if model_type != "ptuning" else f"{plm_name.replace('/', '-')}_{model_type}_{loss_type}_{pooling_method}{is_freeze_bert}_{strategy}_{date}"
output_dir = os.path.join(task_name, output_dir)


# read data
label_col = configParser.get("train_conf", "label_col")
label_list = configParser.get("train_conf", "label_list").split('|')
logger.info(f"label column: {label_col}")
logger.info(f"label list: {label_list}")

logger.info(f"loss type: {loss_type}")


now = datetime.now()


# plm models
pretrain_model = eval(configParser.get("pretrained_models", "pretrain_model"))
print(type(pretrain_model))

model_dict = {
    'ptuning': BertFamily.BertPrefixForSequenceClassification,
    'bert': BertFamily.BertVanillaTextClsForTransformers,
    'BertWithCateFeat': BertFamily.BertClsWithCateFeat,
    'rdrop': BertFamily.RDropBert,
    'deberta': BertFamily.DebertaVanillaTextCls
}



train_set = load_dataset(filepath=trainset_path,
                         label_col=label_col, label_list=label_list)
valid_set = load_dataset(filepath=devset_path,
                         label_col=label_col, label_list=label_list)
test_set = load_dataset(filepath=testset_path,
                        label_col=label_col, label_list=label_list)


shuffle(train_set)


cache_dir = os.path.join('/home/henry/model_zoo', plm_name.split('/')[-1])

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=plm_name,
    cache_dir=cache_dir
)


logger.info(f"Vocab size: {len(tokenizer)}")
if add_vocab:
    import pickle
    with open(vocab_path, 'rb') as f:
        add_vocab_list = pickle.load(f)
    tokenizer.add_tokens(add_vocab_list)
    logger.info(f"Vocab size: {len(tokenizer)}")


train_encodings = tokenizer(train_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length', return_token_type_ids=True)
valid_encodings = tokenizer(valid_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length', return_token_type_ids=True)
test_encodings = tokenizer(test_set[text_col].tolist(
), max_length=max_length, truncation=True, padding='max_length', return_token_type_ids=True)



train_set['model_input'] = train_set[text_col]
valid_set['model_input'] = valid_set[text_col]
test_set['model_input'] = test_set[text_col]

logger.info(f"{'='*20} Input Sample {'='*20}")
for i in range(5):
    logger.info(train_set['model_input'].tolist()[i])
    logger.info(train_encodings['input_ids'][i])
    logger.info(train_set[label_col].tolist()[i])
    logger.info(train_set['one_hot_label'].tolist()[i])
    # if is_cate_feat:
    #     logger.info(train_set['cate_feat_one_hot'].tolist()[i])
logger.info(f"{'='*20} Input Sample {'='*20}")


train_dataset = Dataset(train_set['one_hot_label'].tolist(),
                        train_encodings,
                        train_set['model_input'].tolist())
eval_dataset = Dataset(valid_set['one_hot_label'].tolist(),
                       valid_encodings,
                       valid_set['model_input'].tolist())

test_dataset = Dataset(test_set['one_hot_label'].tolist(),
                       test_encodings,
                       test_set['model_input'].tolist())


label_mat = np.array(train_set['one_hot_label'].tolist())
logger.info(
    "=============================Trainset label info=============================")
label_cnt = np.sum(label_mat, axis=0)
logger.info(f"{label_cnt}")
logger.info(
    "=============================Trainset label info=============================")


if WeightedBCELoss:

    total_label_cnt = np.sum(label_cnt)
    weights = (total_label_cnt / label_cnt).tolist()

else:
    weights = [1] * len(label_list)
logger.info(f"weights: {weights}")


training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    num_train_epochs=epoch_num,  # total number of training epochs
    # batch size per device during training
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,  # batch size for evaluation
    warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    # logging_dir=output_dir,  # directory for storing logs
    # log_level=logging.DEBUG,
    # logging_steps=20,
    learning_rate=lr,
    # lr_scheduler_type='cosine_with_restarts',
    group_by_length=False,
    # eval_accumulation_steps=5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    # evaluation_strategy='steps',
    # eval_steps=20,
    # save_steps=50,
    # save_strategy='steps',
    log_level='debug',
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model='f1',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    # auto_find_batch_size=True,
    greater_is_better=True,
    gradient_checkpointing=gradient_checkpointing
)


# 将日志输出到控制台，默认 sys.stderr
# logger.addHandler(logging.StreamHandler(sys.stdout))
# 输出到文件的格式,注释下面的add',则关闭日志写入
# logger.add(output_dir, level='DEBUG',
#                 format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
#                         "{process.name} | "  # 进程名
#                         "{thread.name} | "  # 进程名
#                         '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
#                 rotation="10 MB")


logger.info('-' * 20)
logger.info('Param \n')
logger.info('Model Type: {}'.format(model_type))
logger.info('Prefix_projection Type: {}'.format(prefix_projection))
logger.info('pre_seq_len: {}'.format(pre_seq_len))
logger.info('Strategy: {}'.format(strategy))


# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=plm_name,
    cache_dir=cache_dir
    # problem_type='multi_label_classification',
    # cache_dir=os.path.join('pretrained_model', model_name.replace('/', '-'), 'config')
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

# train from scratch
model = model_dict[model_type.lower()].from_pretrained(
    plm_name,
    config=config,
    cache_dir=cache_dir
)

# if is_cate_feat:
#     model = model_dict['BertWithCateFeat'].from_pretrained(
#     plm_path,
#     config=config
# )

if add_vocab:
    model.resize_token_embeddings(len(tokenizer))

# continuous training
# model = BertPrefixForSequenceClassification.from_pretrained('/data/haobin/code/few-short-learning-research/slib/result/prefix/vanilla/7/False/output/checkpoint-2550',
    # config=config)

# model = BertPrefixForSequenceClassification.from_pretrained('/expand/Henry/code/AIA_NLP/iSay-NPS/multilabel_cls/roberta_base_ptuning_FocalLoss_weighted_avg_r_drop_2023-05-10_06:47:24/checkpoint-1008',
#                                                             config=config)

trainer = BaseTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    # model_init=model_init,
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=eval_dataset,  # evaluation dataset
    compute_metrics=Metrics.compute_multiclass_metrics if problem_type == 'single_label_classification' else Metrics.compute_multilabel_metrics,
    tokenizer=tokenizer,
    test_key='f1',
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience)],
)


trainer.train()

if trainer.predict:
    label_list = np.array(label_list)
    # predict devset
    predictions, true_labels, _ = trainer.predict(
        eval_dataset, metric_key_prefix="predict")
    # activation = torch.nn.Softmax() if 'single' in problem_type else torch.nn.Sigmoid()
    # preds = activation(torch.Tensor(predictions)).cpu().detach().numpy()
    Metrics.compute_metrics_export_report(pred_logits=predictions, y_true=true_labels, input_df=valid_set,
                                               label_list=label_list, output_dir=output_dir,
                                               prefix='dev', problem_type=problem_type, output_pred_col_name=pred_col_name)
    # predict testset

    predictions, true_labels, _ = trainer.predict(
        test_dataset, metric_key_prefix="predict")
    # preds = activation(torch.Tensor(predictions)).cpu().detach().numpy()
    Metrics.compute_metrics_export_report(pred_logits=predictions, y_true=true_labels, input_df=test_set,
                                               label_list=label_list, output_dir=output_dir,
                                               prefix='test', problem_type=problem_type, output_pred_col_name=pred_col_name, test_with_label=test_with_label)
