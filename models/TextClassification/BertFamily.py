# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/8 15:51
@Author  : Henry.Yuan
@File    : BertTextCls.py

'''

import torch
import sys

from transformers.configuration_utils import PretrainedConfig
from torch import nn
from transformers import BertPreTrainedModel, BertModel, DebertaV2PreTrainedModel, DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from modules.Layers import PrefixEncoder
from modules import Losses
# from pckgs.util import Losses
from torch.utils.checkpoint import checkpoint
import torch.nn as nn

class DebertaVanillaTextCls(DebertaV2PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.config = config
        # self.vanillaBert = VanillaBert(self.config)
        self.dropout = nn.Dropout(p=0.1)
        self.pooling = self.config.pooling
        if self.pooling == 'mean_cat_cls':
            self.classifier = nn.Linear(2*self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'weighted_hidden':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'mean':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        # self.pos_weight = torch.Tensor(config.pos_weight)
        # self.multilable_categorical_loss = config.multilable_categorical_loss
        # self.dice_loss = Losses.BinaryDSCLoss()
        self.weighted_avg = Losses.WeightedLayerPooling(config.num_hidden_layers, layer_start=9)
        self.FocalLoss = self.config.FocalLoss

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        output = self.deberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        last_hidden_state, hidden_states = output.last_hidden_state, output.hidden_states
        

        #
        # idx = torch.arange(token_type_ids.shape[1], 0, -1).cuda()
        # tmp2 = token_type_ids * idx
        # indices = torch.argmax(tmp2, 1, keepdim=True).squeeze()
        # sec_cls = last_hidden_state[torch.arange(last_hidden_state.size(0)), indices]

        # cat_output = torch.cat((pooler_output, 0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls), dim=-1)
        # cat_output = (0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls)
        # cat_output = torch.maximum(last_hidden_state[:, 0, :], sec_cls)
        # cat_output = torch.cat((pooler_output, sec_cls), dim=-1)

        ###################### CLS ##############################
        if self.pooling == 'weighted_hidden':
            weighted_pooling_embeddings = self.weighted_avg(torch.stack(hidden_states))
            weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

            weighted_pooling_embeddings = self.dropout(weighted_pooling_embeddings)
            logits = self.classifier(weighted_pooling_embeddings)
        else:
            mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                              self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
            mean_last_hidden_state = self.dropout(mean_last_hidden_state)
            logits = self.classifier(mean_last_hidden_state)



        

        ####################### weighted layer pooling ###########################
        


        #
        # mask_last_hidden_state = mask_last_hidden_state.float()
        # mask_last_hidden_state[mask_last_hidden_state == 0] = float("-inf")
        # mask_last_hidden_state[mask_last_hidden_state == 1] = 0

        # pooler_output = pooler_output + hidden_states[-2][:, 0, :] + hidden_states[-3][:, 0, :]


        # cat_output = torch.cat((pooler_output, hidden_states[-2][:, 0, :]), dim=-1)


        # weighted avg pooling
        # all_hidden_stats = torch.stack(hidden_states)
        # weighted_pooling_embeddings = self.weighted_avg(all_hidden_stats)
        # weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        # cat_output = self.weighted_avg(last_hidden_state)


        # cat_output = self.dropout(weighted_pooling_embeddings)


        # pooler_output = self.dropout(pooler_output)

        
 
        
        # cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)


        # logits = self.classifier(cat_output)

        # if labels is not None:
        #     random_seed = torch.rand_like(logits).cuda()
        #     mask = torch.where(random_seed >= 0.5, 1, 0)
        #     mask = (labels * mask * 1e12).index_fill(1, torch.LongTensor(self.config.not_negative_sampling_cate).cuda(), 0)
        #     logits = logits + mask

        # logits = self.classifier(output.pooler_output)
        loss = None
        if labels is not None:
            if self.config.problem_type == 'binary_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())
            elif self.config.problem_type == 'single_label_classification':
                if self.FocalLoss:
                    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    # mean over the batch
                    gamma = 2
                    alpha = 0.25
                    loss = (alpha * (1-pt)**gamma * ce_loss).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
            else:
                if self.FocalLoss:
                    loss_fct = Losses.FocalLossV1()
                    # loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())

                # loss_fct = nn.BCEWithLogitsLoss()
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())


                # loss_fct = Losses.NegativeSamplingBCE(self.config.not_negative_sampling_cate)
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())
                # if self.multilable_categorical_loss:
                #     loss = Losses.multilabel_categorical_crossentropy(logits.view(-1, self.config.num_labels),
                #                     labels.view(-1, self.config.num_labels).float())
                # else:
                    # loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss_fct = nn.BCEWithLogitsLoss()
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss = self.dice_loss(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())



            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return logits




class BertVanillaTextCls(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.config = config
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooler_output, last_hidden_state, hidden_states = output.pooler_output, output.last_hidden_state, output.hidden_states
        mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                            self.config.hidden_size) * last_hidden_state
        mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)
        loss = None
        if labels is not None:

            if self.config.problem_type == 'single_label_classification':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())

            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return logits


'''
The Trainer class is optimized for 🤗 Transformers models and can have surprising behaviors when you use it on other models. When using it on your own model, make sure:

your model always return tuples or subclasses of ModelOutput.
your model can compute the loss if a labels argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
your model can accept multiple label arguments (use the label_names in your TrainingArguments to indicate their name to the Trainer) but none of them should be named "label".
'''


class BertVanillaTextClsForTransformers(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.config = config
        # self.vanillaBert = VanillaBert(self.config)
        self.dropout = nn.Dropout(p=0.1)
        self.pooling = self.config.pooling
        if self.pooling == 'mean_cat_cls':
            self.classifier = nn.Linear(2*self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'weighted_hidden':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'mean':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        # self.pos_weight = torch.Tensor(config.pos_weight)
        # self.multilable_categorical_loss = config.multilable_categorical_loss
        # self.dice_loss = Losses.BinaryDSCLoss()
        self.weighted_avg = Losses.WeightedLayerPooling(config.num_hidden_layers, layer_start=9)
        self.FocalLoss = self.config.FocalLoss

        # self.dense_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)



    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooler_output, last_hidden_state, hidden_states = output.pooler_output, output.last_hidden_state, output.hidden_states
        

        #
        # idx = torch.arange(token_type_ids.shape[1], 0, -1).cuda()
        # tmp2 = token_type_ids * idx
        # indices = torch.argmax(tmp2, 1, keepdim=True).squeeze()
        # sec_cls = last_hidden_state[torch.arange(last_hidden_state.size(0)), indices]

        # cat_output = torch.cat((pooler_output, 0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls), dim=-1)
        # cat_output = (0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls)
        # cat_output = torch.maximum(last_hidden_state[:, 0, :], sec_cls)
        # cat_output = torch.cat((pooler_output, sec_cls), dim=-1)

        ###################### CLS ##############################
        if self.pooling == 'mean_cat_cls':
            ####################### CLS cat Mean pooling #############################
            mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                              self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
            
            cat_output = torch.cat((pooler_output, mean_last_hidden_state), dim=-1)
            cat_output = self.dropout(cat_output)
            logits = self.classifier(cat_output)
        elif self.pooling == 'weighted_hidden':
            weighted_pooling_embeddings = self.weighted_avg(torch.stack(hidden_states))
            weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

            weighted_pooling_embeddings = self.dropout(weighted_pooling_embeddings)
            logits = self.classifier(weighted_pooling_embeddings)
        elif self.pooling == 'mean':
            mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                              self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
            mean_last_hidden_state = self.dropout(mean_last_hidden_state)
            logits = self.classifier(mean_last_hidden_state)
        else:
            pooler_output = self.dropout(pooler_output)
            logits = self.classifier(pooler_output)



        

        ####################### weighted layer pooling ###########################
        


        #
        # mask_last_hidden_state = mask_last_hidden_state.float()
        # mask_last_hidden_state[mask_last_hidden_state == 0] = float("-inf")
        # mask_last_hidden_state[mask_last_hidden_state == 1] = 0

        # pooler_output = pooler_output + hidden_states[-2][:, 0, :] + hidden_states[-3][:, 0, :]


        # cat_output = torch.cat((pooler_output, hidden_states[-2][:, 0, :]), dim=-1)


        # weighted avg pooling
        # all_hidden_stats = torch.stack(hidden_states)
        # weighted_pooling_embeddings = self.weighted_avg(all_hidden_stats)
        # weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        # cat_output = self.weighted_avg(last_hidden_state)


        # cat_output = self.dropout(weighted_pooling_embeddings)


        # pooler_output = self.dropout(pooler_output)

        
 
        
        # cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)


        # logits = self.classifier(cat_output)

        # if labels is not None:
        #     random_seed = torch.rand_like(logits).cuda()
        #     mask = torch.where(random_seed >= 0.5, 1, 0)
        #     mask = (labels * mask * 1e12).index_fill(1, torch.LongTensor(self.config.not_negative_sampling_cate).cuda(), 0)
        #     logits = logits + mask

        # logits = self.classifier(output.pooler_output)
        loss = None
        if labels is not None:
            if self.config.problem_type == 'binary_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())
            elif self.config.problem_type == 'single_label_classification':
                if self.FocalLoss:
                    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    # mean over the batch
                    gamma = 2
                    alpha = 0.25
                    loss = (alpha * (1-pt)**gamma * ce_loss).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
            else:
                if self.FocalLoss:
                    loss_fct = Losses.FocalLossV1()
                    # loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())

                # loss_fct = nn.BCEWithLogitsLoss()
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())


                # loss_fct = Losses.NegativeSamplingBCE(self.config.not_negative_sampling_cate)
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())
                # if self.multilable_categorical_loss:
                #     loss = Losses.multilabel_categorical_crossentropy(logits.view(-1, self.config.num_labels),
                #                     labels.view(-1, self.config.num_labels).float())
                # else:
                    # loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss_fct = nn.BCEWithLogitsLoss()
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss = self.dice_loss(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())



            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return logits
        

class BertClsWithCateFeat(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        self.bert = BertModel(config)
        if self.use_gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        self.config = config
        # self.vanillaBert = VanillaBert(self.config)
        self.dropout = nn.Dropout(p=0.1)
        self.pooling = self.config.pooling
        self.cate_feat_hidden_size = self.config.cate_feat_hidden_size
        self.cate_feat_numclass = self.config.cate_feat_numclass

        self.cateEmbedding = nn.Embedding(self.cate_feat_numclass,self.cate_feat_hidden_size)
        if self.pooling == 'mean_cat_cls':
            self.classifier = nn.Linear(2*self.config.hidden_size+self.cate_feat_hidden_size, self.config.num_labels)
        elif self.pooling == 'weighted_hidden':
            self.classifier = nn.Linear(self.config.hidden_size+self.cate_feat_hidden_size, self.config.num_labels)
        elif self.pooling == 'mean':
            self.classifier = nn.Linear(self.config.hidden_size+self.cate_feat_hidden_size, self.config.num_labels)
        elif self.pooling == 'last3avg':
            self.classifier = nn.Linear(3*self.config.hidden_size+self.cate_feat_hidden_size, self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size+self.cate_feat_hidden_size, self.config.num_labels)
        
        # self.pos_weight = torch.Tensor(config.pos_weight)
        # self.multilable_categorical_loss = config.multilable_categorical_loss
        # self.dice_loss = Losses.BinaryDSCLoss()
        self.weighted_avg = Losses.WeightedLayerPooling(config.num_hidden_layers, layer_start=9)
        self.FocalLoss = self.config.FocalLoss
        print(self.cateEmbedding.weight)

        # self.dense_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)



    def forward(self, input_ids, token_type_ids, attention_mask, cate_feat, labels=None, **kwargs):
        # if self.use_gradient_checkpointing:
        #     output = checkpoint(self.bert, input_ids, token_type_ids, attention_mask)
        # else:
        
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooler_output, last_hidden_state, hidden_states = output.pooler_output, output.last_hidden_state, output.hidden_states


        #
        # idx = torch.arange(token_type_ids.shape[1], 0, -1).cuda()
        # tmp2 = token_type_ids * idx
        # indices = torch.argmax(tmp2, 1, keepdim=True).squeeze()
        # sec_cls = last_hidden_state[torch.arange(last_hidden_state.size(0)), indices]

        # cat_output = torch.cat((pooler_output, 0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls), dim=-1)
        # cat_output = (0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls)
        # cat_output = torch.maximum(last_hidden_state[:, 0, :], sec_cls)
        # cat_output = torch.cat((pooler_output, sec_cls), dim=-1)

        ###################### CLS ##############################
        if self.pooling == 'mean_cat_cls':
            ####################### CLS cat Mean pooling #############################
            mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                              self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
            
            bert_embedding = torch.cat((pooler_output, mean_last_hidden_state), dim=-1)
            # cat_output = self.dropout(cat_output)
            # logits = self.classifier(cat_output)
        elif self.pooling == 'weighted_hidden':
            weighted_pooling_embeddings = self.weighted_avg(torch.stack(hidden_states))
            weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

            bert_embedding = self.dropout(weighted_pooling_embeddings)
            # logits = self.classifier(weighted_pooling_embeddings)
        elif self.pooling == 'mean':
            # mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
            #                                                                   self.config.hidden_size) * last_hidden_state
            # bert_embedding = torch.sum(mask_last_hidden_state, dim=1) / (
            #         torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            bert_embedding = sum_embeddings / sum_mask

            # mean_last_hidden_state = self.dropout(mean_last_hidden_state)
            # logits = self.classifier(mean_last_hidden_state)
        elif self.pooling == 'last3avg':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            # bert_embedding = sum_embeddings / sum_mask

            all_hidden_states = torch.stack(hidden_states)
            for i in [-1,-2,-3]:
                cur_hidden_state = all_hidden_states[i, :, :, :]

                sum_embeddings = torch.sum(cur_hidden_state * input_mask_expanded, 1)
                # sum_mask = input_mask_expanded.sum(1)
                # sum_mask = torch.clamp(sum_mask, min=1e-9)
                cur_bert_embedding = sum_embeddings / sum_mask
                bert_embedding = torch.cat((bert_embedding, cur_bert_embedding))


        else:
            bert_embedding = self.dropout(pooler_output)
            # logits = self.classifier(pooler_output)
        if self.use_gradient_checkpointing:
            cate_feat_embedding = checkpoint(self.cateEmbedding, torch.argmax(cate_feat, -1))
        else:
            cate_feat_embedding = self.cateEmbedding(torch.argmax(cate_feat, -1))
        cat_output = torch.cat((bert_embedding, cate_feat_embedding), dim=-1)
        cat_output = self.dropout(cat_output)
        logits = self.classifier(cat_output)




        

        ####################### weighted layer pooling ###########################
        


        #
        # mask_last_hidden_state = mask_last_hidden_state.float()
        # mask_last_hidden_state[mask_last_hidden_state == 0] = float("-inf")
        # mask_last_hidden_state[mask_last_hidden_state == 1] = 0

        # pooler_output = pooler_output + hidden_states[-2][:, 0, :] + hidden_states[-3][:, 0, :]


        # cat_output = torch.cat((pooler_output, hidden_states[-2][:, 0, :]), dim=-1)


        # weighted avg pooling
        # all_hidden_stats = torch.stack(hidden_states)
        # weighted_pooling_embeddings = self.weighted_avg(all_hidden_stats)
        # weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        # cat_output = self.weighted_avg(last_hidden_state)


        # cat_output = self.dropout(weighted_pooling_embeddings)


        # pooler_output = self.dropout(pooler_output)

        
 
        
        # cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)


        # logits = self.classifier(cat_output)

        # if labels is not None:
        #     random_seed = torch.rand_like(logits).cuda()
        #     mask = torch.where(random_seed >= 0.5, 1, 0)
        #     mask = (labels * mask * 1e12).index_fill(1, torch.LongTensor(self.config.not_negative_sampling_cate).cuda(), 0)
        #     logits = logits + mask

        # logits = self.classifier(output.pooler_output)
        loss = None
        if labels is not None:
            if self.config.problem_type == 'binary_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())
            elif self.config.problem_type == 'single_label_classification':
                if self.FocalLoss:
                    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    # mean over the batch
                    gamma = 2
                    # alpha = 0.25
                    alpha = 1 if 'single' in self.config.problem_type else 0.25
                    loss = (alpha * (1-pt)**gamma * ce_loss).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
            else:
                if self.FocalLoss:
                    loss_fct = Losses.FocalLossV1()
                    # loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels).float())

                # loss_fct = nn.BCEWithLogitsLoss()
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())


                # loss_fct = Losses.NegativeSamplingBCE(self.config.not_negative_sampling_cate)
                # loss = loss_fct(logits.view(-1, self.config.num_labels),
                #                 labels.view(-1, self.config.num_labels).float())
                # if self.multilable_categorical_loss:
                #     loss = Losses.multilabel_categorical_crossentropy(logits.view(-1, self.config.num_labels),
                #                     labels.view(-1, self.config.num_labels).float())
                # else:
                    # loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss_fct = nn.BCEWithLogitsLoss()
                    # loss = loss_fct(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())

                    # loss = self.dice_loss(logits.view(-1, self.config.num_labels),
                    #                 labels.view(-1, self.config.num_labels).float())



            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return logits




class RDropBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.config = config
        # self.vanillaBert = VanillaBert(self.config)
        self.dropout = nn.Dropout(p=0.1)
        self.pooling = self.config.pooling
        if self.pooling == 'mean_cat_cls':
            self.classifier = nn.Linear(2*self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'weighted_hidden':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'mean':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        self.loss_fcn = Losses.RDrop(problem_type=self.config.problem_type)
        # self.pos_weight = torch.Tensor(config.pos_weight)
        # self.multilable_categorical_loss = config.multilable_categorical_loss
        # self.dice_loss = Losses.BinaryDSCLoss()
        self.weighted_avg = Losses.WeightedLayerPooling(config.num_hidden_layers, layer_start=9)
        self.FocalLoss = self.config.FocalLoss

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        logits_list = list()
        bert_output_list = list()

        # forward twice
        for i in range(2):
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            pooler_output, last_hidden_state, hidden_states = output.pooler_output, output.last_hidden_state, output.hidden_states


            #
            # idx = torch.arange(token_type_ids.shape[1], 0, -1).cuda()
            # tmp2 = token_type_ids * idx
            # indices = torch.argmax(tmp2, 1, keepdim=True).squeeze()
            # sec_cls = last_hidden_state[torch.arange(last_hidden_state.size(0)), indices]

            # cat_output = torch.cat((pooler_output, 0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls), dim=-1)
            # cat_output = (0.8 * last_hidden_state[:, 0, :] + 0.2 * sec_cls)
            # cat_output = torch.maximum(last_hidden_state[:, 0, :], sec_cls)
            # cat_output = torch.cat((pooler_output, sec_cls), dim=-1)

            ###################### CLS ##############################
            if self.pooling == 'mean_cat_cls':
                ####################### CLS cat Mean pooling #############################
                mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                                self.config.hidden_size) * last_hidden_state
                mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                        torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
                
                cat_output = torch.cat((pooler_output, mean_last_hidden_state), dim=-1)
                cat_output = self.dropout(cat_output)
                logits = self.classifier(cat_output)
            elif self.pooling == 'weighted_hidden':
                weighted_pooling_embeddings = self.weighted_avg(torch.stack(hidden_states))
                weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

                weighted_pooling_embeddings = self.dropout(weighted_pooling_embeddings)
                logits = self.classifier(weighted_pooling_embeddings)
            elif self.pooling == 'mean':
                mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                                self.config.hidden_size) * last_hidden_state
                mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                        torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)
                mean_last_hidden_state = self.dropout(mean_last_hidden_state)
                logits = self.classifier(mean_last_hidden_state)
            else:
                pooler_output = self.dropout(pooler_output)
                logits = self.classifier(pooler_output)

            logits_list.append(logits)
            bert_output_list.append(output)

            #############################
            # output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # pooler_output, last_hidden_state, hidden_states = output.pooler_output, output.last_hidden_state, output.hidden_states
            # cat_output = torch.cat((pooler_output, hidden_states[-2][:, 0, :], hidden_states[-3][:, 0, :]), dim=-1)

            # cat_output = self.dropout(cat_output)

            # logits = self.classifier(cat_output)
            # logits_list.append(logits)
            # bert_output_list.append(output)

        # pad_mask = torch.zeros_like(attention_mask).cuda()
        # pad_mask[attention_mask==0]=1
        # pad_mask[attention_mask==1]=0
        # pad_mask = pad_mask.bool()
        if labels is not None:
            loss = self.loss_fcn(logits_list[0], logits_list[1], labels)
            
        # alpha = 1
        # for logits in logits_list:
        #     if labels is not None:
        #         if self.config.problem_type == 'binary_classification':
        #             loss_fct = nn.BCEWithLogitsLoss()
        #             loss = loss_fct(logits.view(-1, self.config.num_labels),
        #                             labels.view(-1, self.config.num_labels).float())
        #         elif self.config.problem_type == 'single_label_classification':
        #             if self.FocalLoss:
        #                 ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        #                 pt = torch.exp(-ce_loss)
        #                 # mean over the batch
        #                 gamma = 2
        #                 alpha = 0.25
        #                 loss = (alpha * (1-pt)**gamma * ce_loss).mean()
        #             else:
        #                 loss_fct = nn.CrossEntropyLoss()
        #                 loss = loss_fct(logits.view(-1, self.config.num_labels),
        #                                 labels.view(-1, self.config.num_labels).float())
        #         else:
        #             if self.FocalLoss:
        #                 loss_fct = Losses.FocalLossV1()
        #                 # loss_fct = nn.BCEWithLogitsLoss()
        #                 loss = loss_fct(logits.view(-1, self.config.num_labels),
        #                                 labels.view(-1, self.config.num_labels).float())
        #             else:
        #                 loss_fct = nn.BCEWithLogitsLoss()
        #                 loss = loss_fct(logits.view(-1, self.config.num_labels),
        #                                 labels.view(-1, self.config.num_labels).float())

        # if loss is not None:
        #     p, q = logits_list[0], logits_list[1]
        #     if self.config.problem_type == 'single_label_classification':
        #         RDrop
        #     else:
        #         p_loss = F.kl_div(F.logsigmoid(p), F.sigmoid(q), reduction='none')
        #         q_loss = F.kl_div(F.logsigmoid(q), F.sigmoid(p), reduction='none')

        # pad_mask is for seq-level tasks
        # if pad_mask is not None:
        #     p_loss.masked_fill_(pad_mask, 0.)
        #     q_loss.masked_fill_(pad_mask, 0.)


            # p_loss, q_loss = p_loss.mean(), q_loss.mean()
            # kl_loss = (p_loss + q_loss) / 2
            # loss += alpha * kl_loss
            return SequenceClassifierOutput(loss=loss, logits=logits_list[0])
        else:
            return SequenceClassifierOutput(logits=logits_list[0])


class VanillaBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dropout = nn.Dropout(p=0.1)
        self.bert = BertModel(self.config)


    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooler_output, last_hidden_state = outputs.pooler_output, outputs.last_hidden_state
        cat_output = torch.cat([pooler_output, torch.mean(last_hidden_state, dim=1)], dim=-1)
        cat_output = self.dropout(cat_output)
        return cat_output


class FinetuneBertTextCls(nn.Module):
    def __init__(self, config):
        super(FinetuneBertTextCls, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(0.1)
        self.bert = BertModel(self.config)
        if self.config.use_last_hidden_state:
            self.classifier = nn.Linear(3 * self.config.hidden_size, self.config.num_labels)
            # self.last2pooler_weight, self.last3pooler_weight = torch.randn(1, self.config.hidden_size).cuda(), torch.randn(1, self.config.hidden_size).cuda()
            self.last2pooler_weight, self.last3pooler_weight = nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.sigmoid = nn.Sigmoid()
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)



    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output, last_hidden_state, hidden_states = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states


        if self.config.use_last_hidden_state:
            # mean last hidden_state
            mask_last_hidden_state = attention_mask.unsqueeze(-1).expand(-1,-1,self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (torch.sum(attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size)+1e-12)

            # max last hidden_state
            mask_last_hidden_state = mask_last_hidden_state.float()
            mask_last_hidden_state[mask_last_hidden_state==0] = float("-inf")
            mask_last_hidden_state[mask_last_hidden_state==1] = 0
            max_last_hidden_state = torch.max(mask_last_hidden_state+last_hidden_state, dim=1).values
            # max_last_hidden_state = torch.max(last_hidden_state, dim=1).values

            pooler_output = pooler_output + hidden_states[-2][:, 0, :] + hidden_states[-3][:, 0, :]

            cat_output = torch.cat((pooler_output, mean_last_hidden_state, max_last_hidden_state), dim=-1)

        else:
            cat_output = pooler_output

        # print(f"output shape ==> {cat_output.shape}")

        # cat_output = torch.cat([pooler_output, torch.mean(last_hidden_state, dim=1)], dim=-1)

        # cat_output = self.dropout(cat_output)
        logits = self.classifier(cat_output)
        return logits, outputs



class BertPrefixForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.fc_dropout = torch.nn.Dropout(p=0.5)
        if self.config.use_last_hidden_state:
            self.classifier = torch.nn.Linear(3 * config.hidden_size, config.num_labels)
        else:
            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = torch.nn.Sigmoid()

        # for name, param in self.bert.named_parameters():
        #     print(name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        # self.strategy = config.strategy

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()  # [0,1, ...., pre_seq_len] dtype=long
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        print('bert param is {}'.format(bert_param))
        all_param = 0
        for name, param in self.named_parameters():
            # print(name)
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total train param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)  # [B, prompt_len]
        past_key_values = self.prefix_encoder(
            prefix_tokens)  # [B, prompt_len, config.num_hidden_layers * 2 * config.hidden_size]
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # 每组shape = (2, B, #head, pre_seq_len,  n_embd)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)  # (B, pre_seq_len)
        # attention_mask shape (B, max_seq_len)
        prefix_attention_mask = prefix_attention_mask.int()
        sent_attention_mask = attention_mask
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,  # (B,  max_seq_len)
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooler_output, last_hidden_state, hidden_states  = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states

        if self.config.use_last_hidden_state:
            mask_last_hidden_state = sent_attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                         self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                        torch.sum(sent_attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

            # max last hidden_state
            mask_last_hidden_state = mask_last_hidden_state.float()
            mask_last_hidden_state[mask_last_hidden_state == 0] = float("-inf")
            mask_last_hidden_state[mask_last_hidden_state == 1] = 0
            max_last_hidden_state = torch.max(mask_last_hidden_state + last_hidden_state, dim=1).values

            pooler_output = pooler_output + hidden_states[-2][:, 0, :] + hidden_states[-3][:, 0, :]

            cat_output = torch.cat((pooler_output, mean_last_hidden_state, max_last_hidden_state), dim=-1)
            cat_output = self.dropout(cat_output)
            logits = self.classifier(cat_output)
            return logits

        else:
            pooler_output = self.dropout(pooler_output)
            logits = self.classifier(pooler_output)

            return logits
        

# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/20 10:08
@Author  : Henry.Yuan
@File    : ptuningv2.py

'''

# import sys
# sys.path.append("..")
# from util.util import multilabel_categorical_crossentropy


class PTuningV2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.fc_dropout = torch.nn.Dropout(p=0.5)
        # self.classifier = torch.nn.Linear(2*config.hidden_size, self.num_labels)
        # self.classifier = torch.nn.Linear(2*config.hidden_size, self.num_labels)
        # self.classifier = torch.nn.Linear(2*config.hidden_size, self.num_labels)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # self.pos_weight = self.config.pos_weight

        # froze BERT params
        if self.config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.pre_seq_len = self.config.pre_seq_len
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // config.num_attention_heads
        self.strategy = self.config.strategy
        self.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        self.weights = self.config.bce_pos_weight
        self.weighted_bce = self.config.WeightedBCELoss
        self.FocalLoss = self.config.FocalLoss
        # self.weighted_avg = Losses.WeightedLayerPooling(config.num_hidden_layers, layer_start=9)

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()  # [0,1, ...., pre_seq_len] dtype=long

        self.prefix_encoder = PrefixEncoder(self.config)
        self.pooling = self.config.pooling
        if self.pooling == 'mean_cat_cls':
            self.classifier = nn.Linear(2*self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'weighted_hidden':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        elif self.pooling == 'mean':
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # for name, _ in self.named_parameters():
        #     print(name)
        # bert_param = 0
        # for name, param in self.bert.named_parameters():
        #     bert_param += param.numel()
        # # print('bert param is {}'.format(bert_param))
        # all_param = 0
        # for name, param in self.named_parameters():
        #     all_param += param.numel()
        # total_param = all_param - bert_param
        # print('total param is {}'.format(total_param)) 

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        # prefix_tokens.requires_grad = True  # [B, prompt_len]
        if self.use_gradient_checkpointing:
            past_key_values = checkpoint(self.prefix_encoder,
                prefix_tokens)
        else:
        # [B, prompt_len, config.num_hidden_layers * 2 * config.hidden_size]
            past_key_values = self.prefix_encoder(
                prefix_tokens)
        # # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # 每组shape = (2, B, #head, pre_seq_len,  n_embd)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)  # (B, pre_seq_len)
        # attention_mask shape (B, max_seq_len)
        prefix_attention_mask = prefix_attention_mask.int()
        sent_attention_mask = attention_mask
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)


        if self.strategy == 'r_drop':

            loss, logits, outputs = self.r_drop_forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                labels=labels,
                sent_attention_mask=sent_attention_mask
            )
        else:
            loss, logits, outputs = self.bert_cross_entropy(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                labels=labels,
                sent_attention_mask=sent_attention_mask
            )


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


    def bert_cross_entropy(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                           labels, output_attentions, output_hidden_states, past_key_values, return_dict,
                           sent_attention_mask):

        outputs = self.bert(
            input_ids,  # (B,  max_seq_len)
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooler_output, last_hidden_state, hidden_states = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states
        # # pooler_output, _, _ = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states

        # # logits = self.classifier(pooler_output)

        # mask_last_hidden_state = sent_attention_mask.unsqueeze(-1).expand(-1, -1,
        #                                                                   self.config.hidden_size) * last_hidden_state
        # mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
        #         torch.sum(sent_attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

        # input_mask_expanded = sent_attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        # mean_embeddings = sum_embeddings / sum_mask
        
        # cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)
        # cat_output = self.dropout(cat_output)

        # logits = self.classifier(cat_output)

        if self.pooling == 'mean_cat_cls':
            ####################### CLS cat Mean pooling #############################
            # mask_last_hidden_state = sent_attention_mask.unsqueeze(-1).expand(-1, -1,
            #                                                                 self.config.hidden_size) * last_hidden_state
            # mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
            #         torch.sum(sent_attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

            input_mask_expanded = sent_attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)
            cat_output = self.dropout(cat_output)

            logits = self.classifier(cat_output)

        elif self.pooling == 'weighted_hidden':
            weighted_pooling_embeddings = self.weighted_avg(torch.stack(hidden_states))
            weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

            weighted_pooling_embeddings = self.dropout(weighted_pooling_embeddings)
            logits = self.classifier(weighted_pooling_embeddings)
        elif self.pooling == 'mean':
            mask_last_hidden_state = sent_attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                            self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(sent_attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

            input_mask_expanded = sent_attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            logits = self.classifier(mean_embeddings)
        else:
            pooler_output = self.dropout(pooler_output)
            logits = self.classifier(pooler_output)


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.FocalLoss:
                    # important to add reduction='none' to keep per-batch-item loss
                    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    # mean over the batch
                    gamma = 2
                    alpha = 0.25
                    loss = (alpha * (1-pt)**gamma * ce_loss).mean()
                else:

                    loss_fct = nn.CrossEntropyLoss()
                    # loss_fct = Losses.DiceLoss(weight=self.pos_weight)
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                    labels.view(-1, self.config.num_labels))
            elif self.config.problem_type == "multi_label_classification":
                if self.weighted_bce:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    loss = (loss * torch.Tensor(self.weights).cuda()).mean()

                elif self.FocalLoss:
                    loss_fct = Losses.FocalLossV1()
                    loss = loss_fct(logits, labels)
                else:
                    
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels),
                                labels.view(-1, self.config.num_labels).float())
                
        return loss, logits, outputs

    def r_drop_forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                labels, output_attentions, output_hidden_states, past_key_values, return_dict, sent_attention_mask):

        logits_list = []
        outputs_list = []
        for i in range(2):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values
            )
            pooler_output, last_hidden_state, hidden_states = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states
            # pooler_output, _, _ = outputs.pooler_output, outputs.last_hidden_state, outputs.hidden_states

            # logits = self.classifier(pooler_output)

            mask_last_hidden_state = sent_attention_mask.unsqueeze(-1).expand(-1, -1,
                                                                            self.config.hidden_size) * last_hidden_state
            mean_last_hidden_state = torch.sum(mask_last_hidden_state, dim=1) / (
                    torch.sum(sent_attention_mask, dim=-1).unsqueeze(-1).expand(-1, self.config.hidden_size) + 1e-12)

            input_mask_expanded = sent_attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            cat_output = torch.cat((pooler_output, mean_embeddings), dim=-1)

            logits = self.classifier(cat_output)

            logits_list.append(logits)
            outputs_list.append(outputs)

        loss = None
        alpha = 1
        for logits in logits_list:
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    # loss_fct = MSELoss()
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = nn.MSELoss()
                        if loss:
                            loss += alpha * loss_fct(logits.view(-1), labels.view(-1))
                        else:
                            loss = alpha * loss_fct(logits.view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    # loss_fct = CrossEntropyLoss()
                    # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss_fct = nn.CrossEntropyLoss()
                    if loss:
                        loss += 0.5 * loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                    else:
                        loss = 0.5 * loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                elif self.config.problem_type == "multi_label_classification":
                    if self.weighted_bce:
                        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                        raise NotImplementedError("BCEBCEWithLogitsLoss Not Implemented.")
                        # loss = loss_fct(logits, labels)
                        # loss = (loss * torch.Tensor(self.weights).cuda()).mean()
                    elif self.FocalLoss:
                        loss_fct = Losses.FocalLossV1()
                        # loss = loss_fct(logits, labels)
                    else:
                        loss_fct = nn.BCEWithLogitsLoss()
                        # loss = loss_fct(logits.view(-1, self.config.num_labels),
                        #             labels.view(-1, self.config.num_labels).float())


                    # loss_fct = BCEWithLogitsLoss()
                    # loss = loss_fct(logits, labels)
                    if loss:
                        loss += 0.5 * loss_fct(logits, labels)
                    else:
                        loss = 0.5 * loss_fct(logits, labels)

        if loss is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss += 1.0 * loss_fct(logits_list[0].view(-1), logits_list[-1].view(-1))
            else:

                kl_loss = compute_kl_loss(logits_list[0], logits_list[1])
                loss += alpha * kl_loss

        return loss, logits_list[0], outputs_list[0]


def compute_kl_loss(p, q, pad_mask=None):
    # p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    # q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    p_loss = F.kl_div(F.logsigmoid(p), F.sigmoid(q), reduction='none')
    q_loss = F.kl_div(F.logsigmoid(q), F.sigmoid(p), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss