# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/12 15:28
@Author  : Henry.Yuan
@File    : eval_metrics.py

'''

import copy
import pandas as pd
import torchmetrics
import torch
from torchmetrics import Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np

from pckgs.util.tools import cm2df


def cls_metrics_by_torchmetrics(pred):
    num_classes = pred.label_ids.shape[-1]

    recall = torchmetrics.Recall(num_classes=num_classes, average='micro').cuda()
    precision = torchmetrics.Precision(num_classes=num_classes, average='micro').cuda()
    f1score = torchmetrics.F1Score(num_classes=num_classes, average='micro').cuda()

    pred, target = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)

    r = recall(pred, target).item()
    p = precision(pred, target).item()
    f1 = f1score(pred, target).item()

    return {
        'f1': f1,
        'precision': p,
        'recall': r
    }


def cls_report_by_sklearn(pred, target, label_name=None):
    # report = classification_report(y_test, y_pred, output_dict=True)

    report = classification_report(y_true=target, y_pred=pred, zero_division=0, target_names=label_name)
    print(report)
    report = classification_report(y_true=target, y_pred=pred, zero_division=0, target_names=label_name, output_dict=True)

    return report


def p_r_f_by_torchmetrics(device, num_labels, threshold=0.5, reduce='micro', multiclass=False):
    '''
    https://torchmetrics.readthedocs.io/en/latest/pages/classification.html#using-the-multiclass-parameter

    :param device:
    :param num_labels:
    :param threshold:
    :param reduce:
    :return:
    '''
    # precision = Precision(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(f'cuda:{device}')
    # recall = Recall(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(f'cuda:{device}')
    # f1 = F1Score(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(f'cuda:{device}')

    precision = Precision(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(device)
    recall = Recall(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(device)
    f1 = F1Score(num_classes=num_labels, multiclass=multiclass, reduce=reduce, threshold=threshold).to(device)


    return precision, recall, f1


def compute_multilabel_metrics(pred, cls_threshold=0.5):
    labels = pred.label_ids
    preds = pred.predictions
    activation = torch.nn.Sigmoid()
    # activation = torch.nn.Softmax()
    preds = activation(torch.Tensor(preds)).cpu().detach().numpy()

    ############ Multilabel ####################
    # if problem_type == 'multi_label_classification':
    pred_label_one_hot = copy.deepcopy(preds)
    pred_label_one_hot[pred_label_one_hot >= cls_threshold] = 1
    pred_label_one_hot[pred_label_one_hot < cls_threshold] = 0

    ############### Multiclass ######################
    # if problem_type == 'single_label_classification':
    #     arr_max_idx = np.argmax(preds, axis=1)
    #     pred_label_one_hot = np.zeros_like(preds)
    #     pred_label_one_hot[np.arange(pred_label_one_hot.shape[0]), arr_max_idx] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_label_one_hot, zero_division=0, average='macro')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_multiclass_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    # activation = torch.nn.Sigmoid()
    activation = torch.nn.Softmax()
    preds = activation(torch.Tensor(preds)).cpu().detach().numpy()

    ############ Multilabel ####################
    # if problem_type == 'multi_label_classification':
    # pred_label_one_hot = copy.deepcopy(preds)
    # pred_label_one_hot[pred_label_one_hot >= cls_threshold] = 1
    # pred_label_one_hot[pred_label_one_hot < cls_threshold] = 0

    ############### Multiclass ######################
    # if problem_type == 'single_label_classification':
    arr_max_idx = np.argmax(preds, axis=1)
    pred_label_one_hot = np.zeros_like(preds)
    pred_label_one_hot[np.arange(pred_label_one_hot.shape[0]), arr_max_idx] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_label_one_hot, zero_division=0, average='macro')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



def compute_metrics_export_report(pred_logits: np.ndarray, 
                                  y_true: np.ndarray, 
                                  input_df: pd.DataFrame, 
                                  label_list: list, 
                                  output_pred_col_name,
                                  output_dir: str, 
                                  prefix='dev', 
                                  cls_threshold=0.5, 
                                  problem_type='single_label_classification',
                                  test_with_label=True,
                                  is_export=True):
    '''
    params:
    preds: pred prob after Sigmoid of Softmax, (N, C)
    y_true: true class, one-hot, (N, C)
    input_df: Data
    label_list: list of label name
    '''
    
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import torch
    
    print(f"================ Output file to {output_dir} ================")

    if 'single' in problem_type:
        activation = torch.nn.Softmax()
        preds = activation(torch.Tensor(pred_logits)).cpu().detach().numpy()
    else:
        activation = torch.nn.Sigmoid()
        preds = activation(torch.Tensor(pred_logits)).cpu().detach().numpy()


    ################ Multi-label ####################
    label_list  = np.array(label_list)
    input_df.reset_index(inplace=True, drop=True)
    if "single" in problem_type:
        arr_max_idx = np.argmax(preds, axis=1)
        pred_label_one_hot = np.zeros_like(preds)
        pred_label_one_hot[np.arange(pred_label_one_hot.shape[0]), arr_max_idx] = 1

        
        pred_label_idx = pred_label_one_hot.argmax(axis=1)
    
        if test_with_label:
            multiclass_true_label = np.array(input_df['one_hot_label'].tolist()).argmax(axis=1)
            cm = confusion_matrix(multiclass_true_label, pred_label_idx)
            cm_df = cm2df(cm, label_list)
            cm_df.to_csv(os.path.join(output_dir, f'{prefix}_cm.csv'), encoding='utf-8-sig')

    ############### Multi-label ######################
    else:
        pred_label_one_hot = copy.deepcopy(preds)
        pred_label_one_hot[pred_label_one_hot >= cls_threshold] = 1
        pred_label_one_hot[pred_label_one_hot < cls_threshold] = 0
        pred_label_idx = np.argwhere(pred_label_one_hot==1)


    ############# 指标分析 ################
    if test_with_label:
        res = classification_report(y_true=y_true, y_pred=pred_label_one_hot, target_names=label_list)
        print(res)
        res = classification_report(y_true=y_true, y_pred=pred_label_one_hot, target_names=label_list, output_dict=True)
        report_df = pd.DataFrame(res).transpose()
        report_df.to_csv(os.path.join(output_dir, f"{prefix}_cls_report.csv"), encoding='utf-8-sig', index=True)

    if 'single' not in problem_type:
        pred_label_idx = np.zeros_like(preds)

    ########## 推理结果 ###################
    output_d = {output_pred_col_name: list(), f'{output_pred_col_name}_prob': list(), f'{output_pred_col_name}_num_label': list()}
    for label_name in label_list:
        output_d[label_name] = list()
    output_d[f'{output_pred_col_name}_top2_prob_cate'] = list()
    output_d[f'{output_pred_col_name}_top2_prob_diff'] = list()
    print(len(pred_label_idx), len(preds))

    for idx, (pred_l_idx, pred_prob) in enumerate(zip(pred_label_idx, preds)):
        
        if "single" in problem_type:
            # output_d[text_col].append(text)
            pred_l_str = label_list[pred_l_idx]
            # output_d['true'].append(true_l_str)
            output_d[output_pred_col_name].append(pred_l_str)
            # 预测标签的概率
            pred_prob_list = [str(pred_prob[_pred_l_idx]) for _pred_l_idx in pred_l_idx.flatten()]

        else:
            # output_d[text_col].append(text)
            # pred_prob[pred_prob >= cls_threshold] = 1
            # pred_prob[pred_prob < cls_threshold] = 0
            pred_l_idx = np.argwhere(pred_prob>=cls_threshold)

            if pred_l_idx.shape[0] == 0:
                pred_l_idx = np.argmax(pred_prob)
                pred_l_str = label_list[pred_l_idx]
                # 预测标签的概率      
                pred_prob_list = [str(pred_prob[_pred_l_idx]) for _pred_l_idx in pred_l_idx.flatten()]
                output_d[f'{output_pred_col_name}_num_label'].append(1)
            else:
                # print(pred_prob, pred_l_idx,pred_l_idx.flatten(), label_list, label_list[pred_l_idx.flatten()])
                pred_l_str = '|'.join(label_list[pred_l_idx.flatten()])
                # 预测标签的概率      
                pred_prob_list = [str(pred_prob[_pred_l_idx]) for _pred_l_idx in pred_l_idx.flatten()]
                output_d[f'{output_pred_col_name}_num_label'].append(pred_l_idx.shape[0])

            output_d[output_pred_col_name].append(pred_l_str)
        

        output_d[f'{output_pred_col_name}_prob'].append('|'.join(pred_prob_list))

        # 加上个类别输出概率
        for iidx, cur_label in enumerate(label_list):
            output_d[cur_label].append(pred_prob[iidx])
        # 找出概率最接近的两个类和概率差值
        argsort_idx = np.argsort(-pred_prob)
        top1, top2 = argsort_idx[0], argsort_idx[1]
        # print(argsort_idx, top1, top2,)
        output_d[f'{output_pred_col_name}_top2_prob_cate'].append(f'{label_list[top1]}|{label_list[top2]}')
        output_d[f'{output_pred_col_name}_top2_prob_diff'].append(f'{pred_prob[top1]-pred_prob[top2]}')
        if 'single' in problem_type:
            output_d[f'{output_pred_col_name}_num_label'] = [1]*len(output_d[output_pred_col_name])
    # for key, values in output_d.items():
    #     print(key, len(values))

        
    pred_df = pd.DataFrame(output_d)
    
    # print(pred_df.info())
    # print(input_df.info())
    # output_df.to_csv('hh.csv')
    output_df = pd.concat([input_df, pred_df], axis=1)
    # print(output_df.info())
    if is_export:
        output_df.to_csv(os.path.join(output_dir, f'{prefix}_pred_res.csv'), encoding='utf-8-sig', index=False)
    return output_df