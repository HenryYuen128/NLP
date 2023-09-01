# -*- coding: utf-8 -*-
'''
@Time    : 2022/7/12 16:12
@Author  : Henry.Yuan
@File    : CustomTrainer.py

'''



from typing import Dict, OrderedDict

from transformers import Trainer
from modules.AdversarialTraining import FGM
import sys
# import logging
# logger = logging.getLogger(__name__)

# _default_log_level = logging.INFO
# logger.setLevel(_default_log_level)

from loguru import logger
logger.remove()
logger.add(sys.stderr)
import os


class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset=None, test_key="accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })


    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics["eval_" + self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics[
                                "test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_" + self.test_key] = test_metrics["test_" + self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    # def training_step(self, model, inputs):
    #     """
    #     Perform a training step on a batch of inputs.
    #     Subclass and override to inject custom behavior.
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    
    #     fgm = FGM(model, epsilon=0.9, emb_name='embedding')
    
    
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    
    #     loss = self.compute_loss(model, inputs)
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     loss.backward()
    #     fgm.attack()
    #     loss_adv = self.compute_loss(model, inputs)
    #     if self.args.n_gpu > 1:
    #         loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
    #     loss_adv.backward()
    #     fgm.restore()
    
    
    
    #     # loss = self.compute_loss(model, inputs)
    #     #
    #     # loss.backward()
    #     # fgm.attack()
    #     # loss_adv = self.compute_loss(model, inputs)
    #     # loss_adv.backward()
    #     # fgm.restore()
    
    
    #     # if is_sagemaker_mp_enabled():
    #     #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     #     return loss_mb.reduce_mean().detach().to(self.args.device)
    #     #
    #     # with self.compute_loss_context_manager():
    #     #     loss = self.compute_loss(model, inputs)
    #     #
    #     # if self.args.n_gpu > 1:
    #     #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     #
    #     # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #     #     # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #     #     loss = loss / self.args.gradient_accumulation_steps
    #     #
    #     # if self.do_grad_scaling:
    #     #     self.scaler.scale(loss).backward()
    #     # elif self.use_apex:
    #     #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #     #         scaled_loss.backward()
    #     # elif self.deepspeed:
    #     #     # loss gets scaled under gradient_accumulation_steps in deepspeed
    #     #     loss = self.deepspeed.backward(loss)
    #     # else:
    #     #     loss.backward()
    
    #     # fgm.attack()
    #     # with self.compute_loss_context_manager():
    #     #     loss_adv = self.compute_loss(model, inputs)
    #     #
    #     # if self.args.n_gpu > 1:
    #     #     loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
    #     #
    #     # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #     #     # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #     #     loss_adv = loss_adv / self.args.gradient_accumulation_steps
    #     #
    #     # if self.do_grad_scaling:
    #     #     self.scaler.scale(loss_adv).backward()
    #     # elif self.use_apex:
    #     #     with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss:
    #     #         scaled_loss.backward()
    #     # elif self.deepspeed:
    #     #     # loss gets scaled under gradient_accumulation_steps in deepspeed
    #     #     loss_adv = self.deepspeed.backward(loss_adv)
    #     # else:
    #     #     loss_adv.backward()
    #     #
    #     #
    #     # fgm.restore()
    
    #     return loss.detach()
    





class BaseTrainerWithCateVar(Trainer):
    def __init__(self, *args, predict_dataset=None, test_key="accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })


    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics["eval_" + self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics[
                                "test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_" + self.test_key] = test_metrics["test_" + self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # def training_step(self, model, inputs):
    #     """
    #     Perform a training step on a batch of inputs.
    #     Subclass and override to inject custom behavior.
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    
    #     fgm = FGM(model, epsilon=0.9, emb_name='embedding')
    
    
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    

    
    
    #     loss = self.compute_loss(model, inputs)
        
    #     loss.backward()
    #     fgm.attack()
    #     loss_adv = self.compute_loss(model, inputs)
    #     loss_adv.backward()
    #     fgm.restore()
    
    #     return loss.detach()