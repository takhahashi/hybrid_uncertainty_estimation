import torch
import numpy as np
from tqdm import tqdm
import time
from sklearn.decomposition import KernelPCA

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import is_custom_head, unpad_features, pad_scores
from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance,
)
from collections import defaultdict

import logging
import pdb
import copy

log = logging.getLogger()


class UeEstimatorTrustscore:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        trainer = self.cls._trainer

        test_hidden_states, test_answers = self._exctract_features_preds(X)
        eval_results = {"trust_score":[]}
        for hidden_state, answer in zip(test_hidden_states, test_answers):
            diffclass_dist = self._diffclass_euclid_dist(hidden_state, int(answer), self.train_hidden_states)
            sameclass_dist= self._sameclass_euclid_dist(hidden_state, int(answer), self.train_hidden_states)
            if sameclass_dist is None:
                eval_results["trust_score"].append(0.)
            else:
                trust_score = diffclass_dist / (diffclass_dist + sameclass_dist)
                eval_results["trust_score"].append(trust_score)
        return eval_results

    def fit_ue(self, X=None, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        trainer = self.cls._trainer
        model.eval()

        log.info(
            "****************Start calcurating hiddenstate on train dataset **************"
        )
        train_dataloader = trainer.get_train_dataloader()
        hidden_states = []
        labels = []
        for step, inputs in enumerate(train_dataloader):
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
            labels.append(inputs["labels"].to('cpu').detach().numpy().copy())
        hidden_states = np.concatenate(hidden_states)
        labels = np.concatenate(labels)
        labels_hidden_states = defaultdict(list)
        for label, hidden_state in zip(labels, hidden_states):
            labels_hidden_states[int(label)].append(hidden_state)
        self.train_hidden_states = labels_hidden_states
        log.info("**************Done.**********************")

    def _exctract_features_preds(self, X):
        cls = self.cls
        model = self.cls._auto_model
        trainer = self.cls._trainer

        test_dataloader = trainer.get_test_dataloader(X)
        hidden_states = []
        answers = []
        for step, inputs in enumerate(test_dataloader):
            outputs = model(**inputs, output_hidden_states=True)
            pooled_output = outputs[1]
            pooled_output = model.dropout(pooled_output)
            logits = model.classifier(pooled_output).tp('cpu').detach().numpy().copy()

            hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
            answers.append(np.argmax(logits, axis=-1))
        hidden_states = np.concatenate(hidden_states)
        answers = np.concatenate(answers)
        return hidden_states, answers
    
    def _diffclass_euclid_dist(test_hidden_state, test_answer, train_hiddens_labels):
        min_dist = None
        for k, v in train_hiddens_labels.items():
            if int(k) != int(test_answer):
                for train_hidden_state in train_hiddens_labels[int(test_answer)]:
                    dist = np.linalg.norm(test_hidden_state-train_hidden_state)
                    if(min_dist is None or dist < min_dist):
                        min_dist = dist
        return min_dist
    
    def _sameclass_euclid_dist(test_hidden_state, test_answer, train_hiddens_labels):
        min_dist = None
        for k, v in train_hiddens_labels.items():
            if int(k) == int(test_answer):
                for train_hidden_state in train_hiddens_labels[int(test_answer)]:
                    dist = np.linalg.norm(test_hidden_state-train_hidden_state)
                    if(min_dist is None or dist < min_dist):
                        min_dist = dist
        return min_dist