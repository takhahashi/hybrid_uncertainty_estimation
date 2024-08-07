from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np

from transformers.file_utils import add_start_docstrings
from transformers import Trainer, TrainerCallback
from ue4nlp.transformers_regularized import (
    SelectiveTrainer,
    LabelDistributionTrainer,
    ExpEntropyTrainer,
)


def get_trainer(
    model_type: str,  
    use_selective: bool,
    use_sngp: bool,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    metric_fn,
    data_collator=None,
    callbacks=None,
) -> "Trainer":
    training_args.save_total_limit = 1
    training_args.save_steps = 1e5
    training_args.task = 'cls'
    training_args.save_safetensors = False
    if not use_selective and not use_sngp:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    elif use_sngp:
        if use_selective:
            trainer = SelectiveSNGPTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                callbacks=callbacks,
            )
        else:
            trainer = SNGPTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                data_collator=data_collator,
                callbacks=callbacks,
            )
    elif use_selective:
        if training_args.reg_type == 'LabelDistributionLearning':
            trainer = LabelDistributionTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        elif training_args.reg_type == 'ExpEntropyLearning':
            trainer = ExpEntropyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        else:
            trainer = SelectiveTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                data_collator=data_collator,
                callbacks=callbacks,
            )
    return trainer


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArgsWithLossCoefs(TrainingArguments):
    """
    reg_type (:obj:`str`, `optional`, defaults to :obj:`reg-curr`):
        Type of regularization.
    lamb (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        lambda value for regularization.
    margin (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        margin value for metric loss.
    """

    reg_type: Optional[str] = field(
        default="reg-curr", metadata={"help": "Type of regularization."}
    )
    lamb: Optional[float] = field(
        default=0.01, metadata={"help": "lambda value for regularization."}
    )
    margin: Optional[float] = field(
        default=0.05, metadata={"help": "margin value for metric loss."}
    )
    lamb_intra: Optional[float] = field(
        default=0.05, metadata={"help": "lambda intra value for metric loss."}
    )
    unc_threshold: Optional[float] = field(
        default=0.5, metadata={"help": "unc_threshold value for RAU loss."}
    )
    gamma_pos: Optional[float] = field(
        default=1, metadata={"help": "gamma for positive class in asymmetric losses."}
    )
    gamma_neg: Optional[float] = field(
        default=4, metadata={"help": "gamma for negative class in asymmetric losses."}
    )
    coverage: Optional[float] = field(
        default=0.5, metadata={"help": "coverage value for Selective loss."}
    )
    lm: Optional[float] = field(default=1, metadata={"help": "lm in Selective losses."})
    alpha: Optional[float] = field(
        default=4, metadata={"help": "alpha in Selective losses."}
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArgsWithMSDCoefs(TrainingArguments):
    """
    mixup (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Use mixup or not.
    self_ensembling (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Use self-ensembling or not.
    omega (:obj:`float`, `optional`, defaults to :obj:`1.0`):
        mixup sampling coefficient.
    lam1 (:obj:`float`, `optional`, defaults to :obj:`1.0`):
        lambda_1 value for regularization.
    lam2 (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        lambda_2 value for regularization.
    """

    mixup: Optional[bool] = field(default=True, metadata={"help": "Use mixup or not."})
    self_ensembling: Optional[bool] = field(
        default=True, metadata={"help": "Use self-ensembling or not."}
    )
    omega: Optional[float] = field(
        default=1.0, metadata={"help": "mixup sampling coefficient."}
    )
    lam1: Optional[float] = field(
        default=1.0, metadata={"help": "lambda_1 value for regularization."}
    )
    lam2: Optional[float] = field(
        default=0.01, metadata={"help": "lambda_2 value for regularization."}
    )


class HybridModelCallback(TrainerCallback):
    def __init__(self, hb_model, trainer):
        super().__init__()
        self.hb_model = hb_model
        self.trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.hb_model.lsb.update()
        for k, v in self.hb_model.lsb.loss_log.items():
            scaled_loss = self.hb_model.diff_weights[k].to('cpu').detach().numpy().copy() * self.hb_model.scale_weights[k].to('cpu').detach().numpy().copy() * v[-1]
            each_task_loss = v[-1]
            self.trainer.log({f"{k}_scaled_loss": scaled_loss, f"{k}_loss":each_task_loss})

class ExpEntCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        
        self.trainer.log({f"exp_loss":np.mean(self.trainer.exp_loss_list), f"entropy":np.mean(self.trainer.entropy_list)})
        self.trainer.exp_loss_list = []
        self.trainer.entropy_list = []

