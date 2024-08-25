from dataclasses import dataclass, field
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np
from pathlib import Path
import random
import torch
import gpytorch
import hydra
import pickle
import pdb

from utils.utils_wandb import init_wandb, wandb
from utils.classification_models import GPModel

from ue4nlp.text_classifier import TextPredictor

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    ElectraForSequenceClassification,
    EarlyStoppingCallback,
)
from datasets import load_metric, load_dataset
from sklearn.model_selection import train_test_split

from utils.utils_ue_estimator import create_ue_estimator
from utils.utils_data import (
    preprocess_function,
    load_data,
    glue_datasets,
    make_data_similarity,
    task_to_keys,
    simple_collate_fn,
    get_score_range,
    upper_score_dic,
)
from utils.utils_models import create_model
from ue4nlp.transformers_regularized import SelectiveTrainer
from utils.utils_tasks import get_config
from utils.utils_train import get_trainer, TrainingArgsWithLossCoefs, HybridModelCallback

import logging
log = logging.getLogger(__name__)

def compute_metrics(is_regression, metric, label_num, p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(np.round(p.predictions[1] * (label_num - 1))) if is_regression else np.argmax(preds, axis=1)
    
    result = metric.compute(predictions=preds, references=p.label_ids)

    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result
    
def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old


@dataclass
class EncoderArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
    )
    model_type: Optional[str] = field(
        default=None,
    )

def train_eval_gp_model(config, training_args, data_args, work_dir=None):
    log.info(f"config:{config}")
    if config.data.task_name == 'asap':
        encoder_name_or_path = f'/content/drive/MyDrive/workdir/trained_models/bert/classification_raw/asap/prompt_id_{config.data.prompt_id}/fold{config.data.fold}/id0'
    elif config.data.task_name == 'riken':
        encoder_name_or_path = f'/content/drive/MyDrive/workdir/trained_models/bert/classification_raw/riken/{config.data.question_id}_{config.data.prompt_id}_{config.data.score_id}/fold{config.data.fold}/id0'
    args_encoder_model = EncoderArguments(model_name_or_path=encoder_name_or_path, model_type='classification')

    ############### Loading dataset ######################

    log.info("Load dataset.")
    datasets = load_data(config)
    log.info("Done with loading the dataset.")


    if config.data.task_name == 'asap':
        low, high = get_score_range(config.data.task_name, config.data.prompt_id)
        num_labels = high - low + 1
    elif config.data.task_name == 'riken':
        high = upper_score_dic[config.data.prompt_id][config.data.score_id]
        low = 0
        num_labels = high - low + 1
    log.info(f"Number of labels: {num_labels}")

    ################ Loading model #######################
    encoder_model, tokenizer = create_model(num_labels, args_encoder_model, data_args, config.ue, config)

    ################ Preprocessing the dataset ###########
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    sentence2_key = (
        None
        if (config.data.task_name in ["bios", "trustpilot", "jigsaw_race", "sepsis_ethnicity", "asap", "riken"])
        else sentence2_key
    )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    label_to_id = None
    f_preprocess = lambda examples: preprocess_function(
        label_to_id, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )
    datasets = datasets.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if "idx" in datasets.column_names["train"]:
        datasets = datasets.remove_columns("idx")

    ################### Training ####################################
    train_dataset = datasets["train"]
    calibration_dataset = None
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    metric = load_metric(
        "accuracy", keep_in_memory=True, cache_dir=config.cache_dir
    )
    is_regression = False
    metric_fn = lambda p: compute_metrics(is_regression, metric, num_labels, p)

    data_collator = simple_collate_fn
    training_args = update_config(training_args, {'fp16':True})
    training_args.save_total_limit = 1
    trainer = Trainer(
        model=encoder_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
        data_collator=data_collator,
    )
    train_dataloader = trainer.get_train_dataloader()
    hidden_states = []
    labels = []    
    trainer.model.eval()
    for step, inputs in enumerate(train_dataloader):
        outputs = trainer.model(**inputs, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        labels.append(inputs["labels"].to('cpu').detach().numpy().copy())

    hidden_states = np.concatenate(hidden_states)
    labels = np.concatenate(labels)
    train_x = torch.FloatTensor(hidden_states)
    train_y = torch.FloatTensor(labels)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    GPmodel = GPModel(train_x, train_y, likelihood)

    if config.do_train:
        epoch = training_args.num_train_epochs
        GPmodel.train()
        likelihood.train()
        GPmodel.covar_module.base_kernel.lengthscale = np.linalg.norm(train_x[0].numpy() - train_x[1].numpy().T) ** 2 / 2

        optimizer = torch.optim.Adam([
            {'params': GPmodel.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=training_args.learning_rate)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, GPmodel)

        for i in range(epoch):
            optimizer.zero_grad()
            output = GPmodel(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            print('Iter %d/%d - Loss: %.3f lengthscale: %.3f noise: %.3f' % (
                i+1, epoch, loss.item(),
                GPmodel.covar_module.base_kernel.lengthscale.item(),
                GPmodel.likelihood.noise.item()
            ))

        torch.save(GPmodel.state_dict(), work_dir)