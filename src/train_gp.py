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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    max_seq_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        # module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def compute_metrics(is_regression, metric, label_num, p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(np.round(p.predictions[1] * (label_num - 1))) if is_regression else np.argmax(preds, axis=1)
    
    result = metric.compute(predictions=preds, references=p.label_ids)

    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def reset_params(model: torch.nn.Module):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_params(model=layer)


def do_predict_eval(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    train_dataset,
    calibration_dataset,
    eval_metric,
    config,
    work_dir,
    model_dir,
    metric_fn,
    max_len,
):
    eval_results = {}

    true_labels = [example["label"] for example in eval_dataset]
    eval_results["true_labels"] = true_labels

    cls = TextClassifier(
        model,
        tokenizer,
        training_args=config.training,
        trainer=trainer,
        max_len=max_len,
    )

    if config.do_eval:
        if config.ue.calibrate:
            cls.predict(calibration_dataset, calibrate=True)
            log.info(f"Calibration temperature = {cls.temperature}")

        log.info("*** Evaluate ***")

        res = cls.predict(eval_dataset)
        preds, probs = res[:2]
        

        eval_score = eval_metric.compute(predictions=preds, references=true_labels)

        log.info(f"Eval score: {eval_score}")
        eval_results["eval_score"] = eval_score
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()

    if config.do_ue_estimate:
        dry_run_dataset = None

        ue_estimator = create_ue_estimator(
            cls,
            config.ue,
            eval_metric,
            calibration_dataset=calibration_dataset,
            train_dataset=train_dataset,
            cache_dir=config.cache_dir,
            config=config,
        )

        ue_estimator.fit_ue(X=train_dataset, X_test=eval_dataset)

        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / "dev_inference.json"))


def train_eval_glue_model(config, training_args, data_args, work_dir=None):
    log.info(f"config:{config}")
    encoder_model_args = config.encoder_model

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

    encoder_model, tokenizer = create_model(num_labels, encoder_model_args, data_args, config.ue, config)

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

    if config.do_train:
        """
        training_args.warmup_steps = int(
            training_args.warmup_ratio  # TODO:
            * len(train_dataset)
            * training_args.num_train_epochs
            / training_args.train_batch_size
        )
        log.info(f"Warmup steps: {training_args.warmup_steps}")
        training_args.logging_steps = training_args.warmup_steps
        """
        training_args.weight_decay_rate = training_args.weight_decay

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
    if config.do_train:
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
        training_iter = training_args.training.iter_num
        GPmodel.train()
        likelihood.train()
        GPmodel.covar_module.base_kernel.lengthscale = np.linalg.norm(train_x[0].numpy() - train_x[1].numpy().T) ** 2 / 2

        optimizer = torch.optim.Adam([
            {'params': GPmodel.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=training_args.lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, GPmodel)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = GPmodel(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        torch.save(GPmodel.state_dict(), work_dir)

    #################### Predicting##########################

    if config.do_eval or config.do_ue_estimate:
        do_predict_eval(
            model,
            tokenizer,
            trainer,
            eval_dataset,
            train_dataset,
            calibration_dataset,
            metric,
            config,
            work_dir,
            model_args.model_name_or_path,
            metric_fn,
            max_seq_length,
        )


def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old




@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    init_wandb(auto_generated_dir, config)

    args_train = TrainingArgsWithLossCoefs(
        output_dir=auto_generated_dir,
    )
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(task_name=config.data.task_name)
    args_data = update_config(args_data, config.data)


    train_eval_glue_model(config, args_train, args_data, auto_generated_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
