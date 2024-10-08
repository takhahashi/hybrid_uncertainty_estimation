import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from transformers import Trainer
import itertools
from tqdm import trange
import numpy as np
import pickle
import json
import pdb
import os

from utils.classification_models import HybridBert
from . import alpaca_calibrator as calibrator

import logging

log = logging.getLogger("text_classifier")


class TextPredictor:
    def __init__(
        self,
        auto_model,
        bpe_tokenizer,
        max_len=192,
        pred_loader_args={"num_workers": 1},
        pred_batch_size=100,
        training_args=None,
        trainer=None,
        selectivenet=False,
        model_type=None,
    ):
        super().__init__()

        self._auto_model = auto_model
        self._bpe_tokenizer = bpe_tokenizer
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        self._training_args = training_args
        self._trainer = trainer
        self._named_parameters = auto_model.named_parameters
        self.temperature = 1.0
        self._max_len = max_len
        self.selectivenet = selectivenet
        self.model_type = model_type

    @property
    def _bert_model(self):
        return self._auto_model

    @property
    def model(self):
        return self._auto_model

    @property
    def tokenizer(self):
        return self._bpe_tokenizer

    def predict(
        self,
        eval_dataset,
        calibrate=False,
        apply_softmax=True,
        return_preds=True,
        return_vec=False,
    ):
        self._auto_model.eval()

        res = self._trainer.predict(eval_dataset, ignore_keys=['hidden_states'])
        if self.model_type == 'regression':
            pred_score = res[0][0]
            pred_lnvar = res[0][1]
        elif self.model_type == 'normalregression':
            pred_score = res[0]
        elif self.model_type == 'hybrid':
            if res[0][0][0].shape[0] == self._auto_model.num_labels:
                logits = res[0][0]
                reg_output = res[0][1]
            else:
                logits = res[0][1]
                reg_output = res[0][0]

        elif self.model_type == 'classification':
            logits = res[0]
            if isinstance(logits, tuple):
                logits = logits[0]
            if self.selectivenet:
                n_cols = logits.shape[-1]
                n_cls = int((n_cols - 1) / 2)
                logits = logits[:, -n_cls:]
        else:
            raise ValueError(f'{self.model_type} is Invalid model_type')

        if self.model_type == 'classification' or self.model_type == 'hybrid':
            if calibrate:
                labels = [example["label"] for example in eval_dataset]
                calibr = calibrator.ModelWithTempScaling(self._auto_model)
                calibr.scaling(
                    torch.FloatTensor(logits),
                    torch.LongTensor(labels),
                    lr=1e-3,  # TODO:
                    max_iter=100,  # TODO:
                )
                self.temperature = calibr.temperature.detach().numpy()[0]
                self.temperature = np.clip(self.temperature, 0.1, 10)

            logits = np.true_divide(logits, self.temperature)

            if apply_softmax:
                probs = F.softmax(torch.tensor(logits), dim=1).numpy()
            else:
                probs = logits

        hidden_states = []
        if return_vec == True:
            model = self._auto_model
            trainer = self._trainer
            model.eval()

            log.info(
                "****************Start calcurating hiddenstate on train dataset **************"
            )
            eval_dataloader = trainer.get_test_dataloader(eval_dataset)
            hidden_states = []

            for step, inputs in enumerate(eval_dataloader):
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())

            hidden_states = np.concatenate(hidden_states)
            hidden_states = list(hidden_states)
            
        if self.model_type == 'hybrid':
            preds = np.round(reg_output.squeeze() * (self._auto_model.num_labels - 1))
            return [preds, probs] + hidden_states + list(res)
        elif self.model_type == 'classification':
            preds = np.argmax(probs, axis=1)
            return [preds, probs] + hidden_states + list(res)
        elif self.model_type == 'regression':
            preds = np.round(pred_score.squeeze() * (self._auto_model.num_labels - 1))
            lnvar = pred_lnvar
            return [preds, lnvar] + hidden_states + list(res)
        elif self.model_type == 'normalregression':
            preds = np.round(pred_score.squeeze() * (self._auto_model.num_labels - 1))
            return [preds] + hidden_states + list(res)