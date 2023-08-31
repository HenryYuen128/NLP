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


class NegativeSamplingBCE(nn.Module):
    def __init__(self, cates, threshold=0.65):
        super(NegativeSamplingBCE, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()
        self.cates = torch.LongTensor(cates).cuda()
        self.threshold = threshold

    def forward(self, out, label):
        random_seed = torch.rand_like(out).cuda()
        mask = torch.where(random_seed >= self.threshold, 1, 0)
        mask = (label * mask * 1e12).index_fill(1, self.cates, 0)
        out = out + mask
        loss = self.loss_fcn(out, label)
        return loss
    
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
def multilabel_categorical_crossentropy(y_pred, y_true):
    """

            :param y_true:
            :param y_pred:
            :return:
            多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
        1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
        不用加激活函数，尤其是不能加sigmoid或者softmax！预测
        阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
        本文。

    """

    y_pred = (1 - 2 * y_true) * y_pred  # 将正例乘以-1，负例乘以1
    y_pred_neg = y_pred - y_true * 1e12  # 将正例变为负无穷，消除影响
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # 将负例变为负无穷
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)  # 0阈值
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    return loss.mean()


class BinaryDSCLoss(torch.nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(BinaryDSCLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss



class WeightedLayerPooling(nn.Module):
    '''
    weighted average

    '''
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
    
class BinaryDiceLoss(nn.Module):
    """
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        assert output.shape == target.shape, 'output & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        self.weight = torch.Tensor(self.weight)

        if target.is_cuda and not self.weight.is_cuda:
            self.weight = self.weight.cuda()

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += (dice_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class multiclass_focal_loss(nn.Module):
    """
    需要保证每个batch的长度一样，不然会报错。
    """
    def __init__(self,alpha=0.25,gamma = 2, num_classes = 2, size_average =True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi) = -α(1-yi)**γ * log(yi)
        :param alpha:
        :param gamma:
        :param num_classes:
        :param size_average:
        """
 
        super(multiclass_focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            print("Focal_loss alpha = {},对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha<1 #如果α为一个常数,则降低第一类的影响
            print("--- Focal_loss alpha = {},将对背景类或者大类负样本进行权重衰减".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma
 
    def forward(self, preds,labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]  B:batch N:检测框数目 C:类别数
        :param labels: 实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds,dim=1)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        # preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        # preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
 
        preds_softmax = preds_softmax.gather(1,labels)
        preds_logsoft = preds_logsoft.gather(1,labels)

        # self.alpha = self.alpha.gather(0,labels.view(-1))
        self.alpha = self.alpha.gather(0,labels)
        loss = -torch.mul(torch.pow((1-preds_softmax),self.gamma),preds_logsoft)
        loss = torch.mul(self.alpha,loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss2d(nn.modules.loss._WeightedLoss):
####################################################
##### This is focal loss class for multi class #####
####################################################
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss



class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """
    def __init__(self, problem_type='single_label_classification'):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.problem_type = problem_type

    def forward(self, logits1, logits2, target, pad_mask=None, kl_weight=1.):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        if self.problem_type == 'single_label_classification':
            ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
            kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1))
            kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1))
            if pad_mask is not None:
                kl_loss1.masked_fill_(pad_mask, 0.)
                kl_loss2.masked_fill_(pad_mask, 0.)
            kl_loss1 = kl_loss1.sum(-1)
            kl_loss2 = kl_loss2.sum(-1)
            kl_loss = (kl_loss1 + kl_loss2) / 2
            loss = ce_loss + kl_weight * kl_loss
        else:
            ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
            kl_loss1 = self.kld(F.logsigmoid(logits1, dim=-1), F.sigmoid(logits2, dim=-1)).sum(-1)
            kl_loss2 = self.kld(F.logsigmoid(logits2, dim=-1), F.softmsigmoidax(logits1, dim=-1)).sum(-1)
            if pad_mask:
                kl_loss1.masked_fill_(pad_mask, 0.)
                kl_loss2.masked_fill_(pad_mask, 0.)
            kl_loss1 = kl_loss1.sum(-1)
            kl_loss2 = kl_loss2.sum(-1)
            kl_loss = (kl_loss1 + kl_loss2) / 2
            loss = ce_loss + kl_weight * kl_loss
        return loss.mean()


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()