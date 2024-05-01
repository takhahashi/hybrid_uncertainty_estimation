from ue4nlp.transformers_cached import (
    ElectraForSequenceClassificationCached,
    ElectraForSequenceClassificationAllLayers,
    BertForSequenceClassificationCached,
    RobertaForSequenceClassificationCached,
    DebertaForSequenceClassificationCached,
    DistilBertForSequenceClassificationCached,
)

from utils.utils_heads import (
    ElectraClassificationHeadCustom,
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraClassificationHeadSN,
    spectral_normalized_model,
    SpectralNormalizedBertPooler,
    SpectralNormalizedPooler,
    ElectraSelfAttentionStochastic,
    replace_attention,
    ElectraClassificationHS,
    BERTClassificationHS,
    SelectiveNet,
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    DebertaForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    BertModel,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
)
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from torch.nn.utils import spectral_norm
import torch
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from safetensors.torch import load_file

log = logging.getLogger(__name__)


def build_model(model_class, model_path_or_name, **kwargs):
    return model_class.from_pretrained(model_path_or_name, **kwargs)


def load_electra_sn_encoder(model_path_or_name, model):

    with open(model_path_or_name + "/model.safetensors", "rb") as f:
        data = f.read()
    model_full = torch.load(data)

    for i, electralayer in enumerate(model.electra.encoder.layer):
        electralayer_name = f"electra.encoder.layer.{i}.output.dense"
        electralayer.output.dense.weight_orig.data = model_full[
            f"{electralayer_name}.weight_orig"
        ].data
        electralayer.output.dense.weight_u.data = model_full[
            f"{electralayer_name}.weight_u"
        ].data
        electralayer.output.dense.weight_v.data = model_full[
            f"{electralayer_name}.weight_v"
        ].data
        electralayer.output.dense.bias.data = model_full[
            f"{electralayer_name}.bias"
        ].data

    del model_full
    torch.cuda.empty_cache()
    log.info("Loaded Electra's SN encoder")


def load_electra_sn_head(model_path_or_name, model, name):
    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
    model.classifier.eval_init(model_full)
    del model_full
    torch.cuda.empty_cache()
    log.info(f"Loaded {name}'s head")


def load_distilbert_sn_head(model_path_or_name, model):
    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
    model.pre_classifier.weight_orig.data = model_full[
        "pre_classifier.weight_orig"
    ].data
    model.pre_classifier.weight_u.data = model_full["pre_classifier.weight_u"].data
    model.pre_classifier.weight_v.data = model_full["pre_classifier.weight_v"].data
    model.pre_classifier.bias.data = model_full["pre_classifier.bias"].data
    del model_full
    torch.cuda.empty_cache()
    log.info("Loaded DistilBERT's head")


def load_bert_sn_pooler(model_path_or_name, model):
    model_full = load_file(model_path_or_name + "/model.safetensors")
    model.bert.pooler.dense.weight_orig.data = model_full[
        "bert.pooler.dense.weight_orig"
    ].data
    model.bert.pooler.dense.weight_u.data = model_full[
        "bert.pooler.dense.weight_u"
    ].data
    model.bert.pooler.dense.weight_v.data = model_full[
        "bert.pooler.dense.weight_v"
    ].data
    model.bert.pooler.dense.bias.data = model_full["bert.pooler.dense.bias"].data
    del model_full
    torch.cuda.empty_cache()
    log.info("Loaded BERT's SN pooler")


def create_bert(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if ue_args.use_cache:
        if use_sngp:
            model_kwargs.update(dict(ue_config=ue_args.sngp))
            model = build_model(
                SNGPBertForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
        elif use_spectralnorm:
            model = build_model(
                BertForSequenceClassificationCached, model_path_or_name, **model_kwargs
            )
            model.use_cache = True
            model.bert.pooler = SpectralNormalizedBertPooler(model.bert.pooler)
            log.info("Replaced BERT Pooler with SN")
            if config.do_eval and not (config.do_train):
                load_bert_sn_pooler(model_path_or_name, model)
        else:
            # common BERT case
            model = build_model(
                BertForSequenceClassificationCached, model_path_or_name, **model_kwargs
            )
            if ("use_hs_labels" in ue_args.keys()) and ue_args.use_hs_labels:
                model.classifier = BERTClassificationHS(
                    model.classifier, n_labels=model_config.num_labels
                )
                log.info("Replaced BERT's head with hyperspherical labels")
        model.disable_cache()
    else:
        # without cache
        if use_spectralnorm and not (use_mixup):
            model = build_model(
                BertForSequenceClassification, model_path_or_name, **model_kwargs
            )
            model.use_cache = False
            model.bert.pooler = SpectralNormalizedBertPooler(model.bert.pooler)
            log.info("Replaced BERT Pooler with SN")
            if config.do_eval and not (config.do_train):
                load_bert_sn_pooler(model_path_or_name, model)
            """
        elif config.data.task_name == 'asap' or config.data.task_name == 'riken':
            if model_config.inf_type == 'regression':
                raise ValueError(f"regression regression regression regression")
            else:
                model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
                )
                model.load_state_dict(torch.load(config.model.model_path))
            """ 
        elif model_config.hybrid:
            model = build_model(
                HybridBert, model_path_or_name, **model_kwargs
            )

        else:
            model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
            )
    return model


def create_fairlib_bert(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    """
    Loads pretrained BERT model from fairlib.
    Imports are defined inside function to avoid unwanted installations.
    """
    from fairlib import BaseOptions
    from ue4nlp.models_fairlib import (
        BertForSequenceClassificationFairlib,
        BertForSequenceClassificationFairlibINLP,
    )

    # get model parameters
    fairlib_model_args = config.get("fairlib", None)
    options = BaseOptions()
    state = options.get_state(args=fairlib_model_args, silence=True)
    # build a fairlib model with params
    if "INLP_checkpoint" in fairlib_model_args.keys():
        model = BertForSequenceClassificationFairlibINLP(state)
    else:
        model = BertForSequenceClassificationFairlib(state)
    # print(model)
    # load model from checkpoint
    log.info(os.getcwd())
    model_path = os.path.join(model_path_or_name, fairlib_model_args.checkpoint_path)
    map_to_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=map_to_device)
    model.load_state_dict(state_dict["model"])
    # add config for compatibility
    model.config = model_config
    if "INLP_checkpoint" in fairlib_model_args.keys():
        INLP_checkpoint = os.path.join(
            model_path_or_name, fairlib_model_args.INLP_checkpoint
        )
        states = torch.load(INLP_checkpoint, map_location="cuda:0")
        model._post_init(states["classifier"], states["P"])
    return model


def create_fairlib_mlp(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    """
    Loads pretrained MLP model from fairlib.
    Imports are defined inside function to avoid unwanted installations.
    """
    from fairlib import BaseOptions
    from ue4nlp.models_fairlib import (
        MLPForSequenceClassificationFairlib,
        MLPForSequenceClassificationFairlibINLP,
    )

    # get model parameters
    fairlib_model_args = config.get("fairlib", None)
    options = BaseOptions()
    state = options.get_state(args=fairlib_model_args, silence=True)
    # build a fairlib model with params
    if "INLP_checkpoint" in fairlib_model_args.keys():
        model = MLPForSequenceClassificationFairlibINLP(state)
    else:
        model = MLPForSequenceClassificationFairlib(state)
    # load model from checkpoint
    log.info(os.getcwd())
    model_path = os.path.join(model_path_or_name, fairlib_model_args.checkpoint_path)
    # here we don't check spectral norm, cause we already create SN model using fairlib state
    map_to_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=map_to_device)
    model.load_state_dict(state_dict["model"])
    # add config for compatibility
    model.config = model_config
    if "INLP_checkpoint" in fairlib_model_args.keys():
        INLP_checkpoint = os.path.join(
            model_path_or_name, fairlib_model_args.INLP_checkpoint
        )
        states = torch.load(INLP_checkpoint)
        model.post_init(states["classifier"], states["P"])
    else:
        model.post_init()
    return model


def create_electra(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    # TODO: rearrange if
    if ue_args.ue_type == "l-maha" or ue_args.ue_type == "l-nuq":
        electra_classifier = ElectraForSequenceClassificationAllLayers
    else:
        electra_classifier = ElectraForSequenceClassification
    if ue_args.use_cache:
        if use_sngp:
            model_kwargs.update(dict(ue_config=ue_args.sngp))
            model = build_model(
                SNGPElectraForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            log.info("Loaded ELECTRA with SNGP")
        elif use_spectralnorm:
            model = build_model(
                ElectraForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            if "last" in config.spectralnorm_layer:
                sn_value = (
                    None if "sn_value" not in ue_args.keys() else ue_args.sn_value
                )
                n_power_iterations = (
                    1
                    if "n_power_iterations" not in ue_args.keys()
                    else ue_args.n_power_iterations
                )
                model.classifier = ElectraClassificationHeadSN(
                    model.classifier, sn_value, n_power_iterations
                )
                log.info("Replaced ELECTRA's head with SN")
                if (
                    config.do_eval
                    and not (config.do_train)
                    and not (config.ue.reg_type == "selectivenet")
                ):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            elif config.spectralnorm_layer == "all":
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")
        else:
            model = build_model(
                ElectraForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            if ("use_hs_labels" in ue_args.keys()) and ue_args.use_hs_labels:
                if not os.path.exists(Path(model_path_or_name) / "hs_labels.pt"):
                    hs_labels = None
                else:
                    hs_labels = torch.load(Path(model_path_or_name) / "hs_labels.pt")
                model.classifier = ElectraClassificationHS(
                    model.classifier,
                    n_labels=model_config.num_labels,
                    hs_labels=hs_labels,
                )
                torch.save(
                    model.classifier.hs_labels, Path(config.output_dir) / "hs_labels.pt"
                )
                log.info("Replaced ELECTRA's head with hyperspherical labels")
            else:
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                log.info("Replaced ELECTRA's head")
        model.disable_cache()
        if ue_args.get("use_sto", False):
            # replace attention by stochastic version
            if ue_args.sto_layer == "last":
                model = replace_attention(model, ue_args, -1)
            elif ue_args.sto_layer == "all":
                for idx, _ in enumerate(model.electra.encoder.layer):
                    model = replace_attention(model, ue_args, idx)
            log.info("Replaced ELECTRA's attention with Stochastic Attention")
    else:
        if use_duq:
            log.info("Using ELECTRA DUQ model")
            model = build_model(
                ElectraForSequenceClassificationDUQ, model_path_or_name, **model_kwargs
            )
            model.make_duq(
                output_dir=config.cache_dir,
                batch_size=config.training.per_device_train_batch_size,
                duq_params=config.ue.duq_params,
            )
        elif use_spectralnorm and not (use_mixup):
            model = build_model(electra_classifier, model_path_or_name, **model_kwargs)
            if "last" in config.spectralnorm_layer:
                sn_value = (
                    None if "sn_value" not in ue_args.keys() else ue_args.sn_value
                )
                n_power_iterations = (
                    1
                    if "n_power_iterations" not in ue_args.keys()
                    else ue_args.n_power_iterations
                )
                model.classifier = ElectraClassificationHeadSN(
                    model.classifier, sn_value, n_power_iterations
                )
                log.info("Replaced ELECTRA's head with SN")
                if (
                    config.do_eval
                    and not (config.do_train)
                    and not (config.ue.reg_type == "selectivenet")
                ):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            elif config.spectralnorm_layer == "all":
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                spectral_normalized_model(model)
                log.info("Replaced ELECTRA's encoder with SN")
        elif use_mixup:
            model = build_model(
                ElectraForSequenceClassificationMSD, model_path_or_name, **model_kwargs
            )
            # set MSD params
            log.info("Created mixup model")
            model.post_init(config.mixup)
            if use_spectralnorm:
                if config.spectralnorm_layer == "last":
                    model.classifier = ElectraClassificationHeadSN(model.classifier)
                    if model.self_ensembling:
                        model.model_2.classifier = ElectraClassificationHeadSN(
                            model.model_2.classifier
                        )
                    log.info("Replaced ELECTRA's head with SN")
                elif config.spectralnorm_layer == "all":
                    model.classifier = ElectraClassificationHeadCustom(model.classifier)
                    if model.self_ensembling:
                        model.model_2.classifier = ElectraClassificationHeadCustom(
                            model.model_2.classifier
                        )
                    spectral_normalized_model(model)
                    log.info("Replaced ELECTRA's encoder with SN")
                if (
                    config.do_eval
                    and not (config.do_train)
                    and not (config.ue.reg_type == "selectivenet")
                ):
                    load_electra_sn_head(model_path_or_name, model, "ELECTRA SN")
            else:
                # TODO: Check how this works if we replaced both classifiers
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                if model.self_ensembling:
                    model.model_2.classifier = ElectraClassificationHeadCustom(
                        model.model_2.classifier
                    )
                log.info("Replaced ELECTRA's head")
        else:
            model = build_model(electra_classifier, model_path_or_name, **model_kwargs)
            # model.classifier = ElectraClassificationHeadCustom(model.classifier)
            if ("use_hs_labels" in ue_args.keys()) and ue_args.use_hs_labels:
                if not os.path.exists(Path(model_path_or_name) / "hs_labels.pt"):
                    hs_labels = None
                else:
                    hs_labels = torch.load(Path(model_path_or_name) / "hs_labels.pt")
                model.classifier = ElectraClassificationHS(
                    model.classifier,
                    n_labels=model_config.num_labels,
                    hs_labels=hs_labels,
                )
                torch.save(
                    model.classifier.hs_labels, Path(config.output_dir) / "hs_labels.pt"
                )
                log.info("Replaced ELECTRA's head with hyperspherical labels")
            else:
                model.classifier = ElectraClassificationHeadCustom(model.classifier)
                log.info("Replaced ELECTRA's head")
            log.info("Replaced ELECTRA's head")
        if ue_args.get("use_sto", False):
            # replace attention by stochastic version
            if ue_args.sto_layer == "last":
                model = replace_attention(model, ue_args, -1)
            elif ue_args.sto_layer == "all":
                for idx, _ in enumerate(model.electra.encoder.layer):
                    model = replace_attention(model, ue_args, idx)
            log.info("Replaced ELECTRA's attention with Stochastic Attention")

    if config.ue.reg_type == "selectivenet":
        model.classifier = SelectiveNet(model.classifier)
        if config.do_eval and not (config.do_train):
            model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
            model.classifier.eval_init(model_full)
            del model_full
            torch.cuda.empty_cache()
            log.info(f"Loaded ELECTRA's SelectiveNet head")

    if use_spectralnorm and "resid" in config.spectralnorm_layer:
        if "last" not in config.spectralnorm_layer:
            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")
        for electralayer in model.electra.encoder.layer:
            electralayer.output.dense = torch.nn.utils.spectral_norm(
                electralayer.output.dense
            )
        log.info("Replaced residual connections after attention in ELECTRA's encoder")
        if config.do_eval and not (config.do_train):
            load_electra_sn_encoder(model_path_or_name, model)

    return model


def create_roberta(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_spectralnorm:
        if ue_args.use_cache:
            model = build_model(
                RobertaForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(
                RobertaForSequenceClassification, model_path_or_name, **model_kwargs
            )
            model.use_cache = False
        model.classifier = ElectraClassificationHeadSN(model.classifier)
        log.info("Replaced RoBERTA's head with SN")
        if config.do_eval and not (config.do_train):
            load_electra_sn_head(model_path_or_name, model, "RoBERTA SN")
    else:
        if ue_args.use_cache:
            model = build_model(
                RobertaForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(
                RobertaForSequenceClassification, model_path_or_name, **model_kwargs
            )
            model.use_cache = False
        model.classifier = ElectraClassificationHeadCustom(model.classifier)
        log.info("Replaced RoBERTA's head")
    return model


def create_distilroberta(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_spectralnorm:
        if ue_args.use_cache:
            model = build_model(
                RobertaForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
            )
            model.use_cache = False
        model.classifier = ElectraClassificationHeadSN(model.classifier)
        log.info("Replaced DisitlRoBERTA's head with SN")
        if config.do_eval and not (config.do_train):
            load_electra_sn_head(model_path_or_name, model, "DisitlRoBERTA SN")
    else:
        if ue_args.use_cache:
            model = build_model(
                RobertaForSequenceClassificationCached,
                model_path_or_name,
                **model_kwargs,
            )
            model.use_cache = True
            model.disable_cache()
        else:
            model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
            )
            model.use_cache = False
        model.classifier = ElectraClassificationHeadCustom(model.classifier)
        log.info("Replaced DisitlRoBERTA's head")
    return model


def create_deberta(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(
            DebertaForSequenceClassificationMSD, model_path_or_name, **model_kwargs
        )
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pooler = SpectralNormalizedPooler(model.pooler)
                if model.self_ensembling:
                    model.model_2.pooler = SpectralNormalizedPooler(
                        model.model_2.pooler
                    )
                log.info("Replaced DeBERTA's pooler with SN")
                if config.do_eval and not (config.do_train):
                    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
                    model.pooler.eval_init(model_full)
                    del model_full
                    log.info("Loaded DeBERTA's pooler with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
    else:
        # build cached model with or without cache, and add spectralnorm case
        model = build_model(
            DebertaForSequenceClassificationCached, model_path_or_name, **model_kwargs
        )
        model.use_cache = True if ue_args.use_cache else False
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                sn_value = (
                    None if "sn_value" not in ue_args.keys() else ue_args.sn_value
                )
                model.pooler = SpectralNormalizedPooler(model.pooler, sn_value)
                log.info("Replaced DeBERTA's pooler with SN")
                if config.do_eval and not (config.do_train):
                    model_full = torch.load(model_path_or_name + "/pytorch_model.bin")
                    model.pooler.eval_init(model_full)
                    del model_full
                    log.info("Loaded DeBERTA's pooler with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DeBERTA's encoder with SN")
        elif not ue_args.use_cache:
            model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
            )
        if ue_args.use_cache:
            model.disable_cache()
    return model


def create_distilbert(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(
            DistilBertForSequenceClassificationMSD, model_path_or_name, **model_kwargs
        )
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pre_classifier = spectral_norm(model.pre_classifier)
                if model.self_ensembling:
                    model.model_2.pre_classifier = spectral_norm(
                        model.model_2.pre_classifier
                    )
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
            if config.do_eval and not (config.do_train):
                load_distilbert_sn_head(model_path_or_name, model)
    else:
        model = build_model(
            DistilBertForSequenceClassificationCached,
            model_path_or_name,
            **model_kwargs,
        )
        model.use_cache = True if ue_args.use_cache else False
        if use_spectralnorm:
            if config.spectralnorm_layer == "last":
                model.pre_classifier = spectral_norm(model.pre_classifier)
                log.info("Replaced DistilBERT's head with SN")
            elif config.spectralnorm_layer == "all":
                spectral_normalized_model(model)
                log.info("Replaced DistilBERT's encoder with SN")
            if config.do_eval and not (config.do_train):
                load_distilbert_sn_head(model_path_or_name, model)
        elif not ue_args.use_cache:
            model = build_model(
                AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
            )
        if ue_args.use_cache:
            model.disable_cache()
    return model


def create_xlnet(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if use_mixup:
        model = build_model(
            XLNetForSequenceClassificationMSD, model_path_or_name, **model_kwargs
        )
        # set MSD params
        log.info("Created mixup model")
        model.post_init(config.mixup)
        # model.classifier = ElectraClassificationHeadCustom(model.classifier)
        log.info("Don't replaced XLNet's head")
    else:
        model = build_model(
            AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
        )
    return model

def create_hybridbert(
    model_config,
    tokenizer,
    use_sngp,
    use_duq,
    use_spectralnorm,
    use_mixup,
    use_selective,
    ue_args,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if 'hybridbert' == model_path_or_name:
        model_path_or_name = 'bert-base-uncased'
    model = build_model(
        HybridBert, model_path_or_name, **model_kwargs
    )
    return model

@dataclass
class HybridOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    reg_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class HybridBert(BertForSequenceClassification):
    def __init__(self, config):
        super(HybridBert, self).__init__(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.lsb = ScaleDiffBalance(task_names=['regression', 'classification'])

        nn.init.normal_(self.regressor.weight, std=0.02)  # 重みの初期化
        nn.init.normal_(self.regressor.bias, 0)
    
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.inference_body(
            body=self.bert,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        regressor_output = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            ########classification loss#########
            loss_fct = CrossEntropyLoss()
            c_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            ########regression loss########
            reg_labels = labels.view(-1) / (self.num_labels - 1)
            loss_fct = MSELoss()
            r_loss = loss_fct(regressor_output.view(-1), reg_labels)
            
            loss, s_wei, diff_wei, alpha, pre_loss = self.lsb(regression=r_loss, classification=c_loss)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return HybridOutput(
            loss=loss,
            logits=logits,
            reg_output=regressor_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
class ScaleDiffBalance:
  def __init__(self, task_names, priority=None, beta=1.):
    self.task_names = task_names
    self.num_tasks = len(self.task_nemes)
    self.task_priority = {}
    if priority is not None:
        for k, v in priority.items():
           self.task_priority[k] = v
    else:
        for k in self.task_names:
          self.task_priority[k] =  1/self.num_tasks
    self.all_loss_log = []
    self.loss_log = defaultdict(list)
    self.beta = beta
    self.all_batch_loss = 0
    for k in self.task_names:
        self.each_task_batch_loss[k] = 0
    self.batch_count = 0
  
  def update(self, *args, **kwargs):
    self.all_loss_log = np.append(self.all_loss_log, self.all_batch_loss/self.batch_count)
    self.all_batch_loss = 0
    for k, v in self.each_task_batch_loss.items():
       self.loss_log[k] = np.append(self.loss_log[k], v)
       self.each_task_batch_loss[k] = 0
    self.batch_count = 0
  
  def __call__(self, *args, **kwargs):
    self.batch_count += 1
    scale_weights = self._calc_scale_weights()
    diff_weights = self._calc_diff_weights()
    alpha = self._calc_alpha(diff_weights)
    all_loss = 0
    for k, each_loss in kwargs:
       all_loss += scale_weights[k] * diff_weights[k] * each_loss
       self.each_task_batch_loss[k] += each_loss
    if len(self.all_loss_log) < 1:
      pre_loss = 0
    else:
      pre_loss = self.all_loss_log[-1]
    self.all_batch_loss += alpha * all_loss
    return alpha * all_loss, scale_weights, diff_weights, alpha, pre_loss
  
  def _calc_scale_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 1:
      for k, v in self.task_priority.items():
         w_dic[k] = v.cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         task_priority = self.task_priority[k]
         w_dic[k] = torch.tensor(self.all_loss_log[-1]*task_priority/each_task_loss_arr[-1]).cuda()
    return w_dic
  
  def _calc_diff_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 2:
      for k, _ in self.task_priority.items():
         w_dic[k] = torch.tensor(1.).cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         w_dic[k] = torch.tensor(((each_task_loss_arr[-1]/each_task_loss_arr[-2])/(self.all_loss_log[-1]/self.all_loss_log[-2]))**self.beta).cuda()
    return w_dic
  
  def _calc_alpha(self, diff_weights):
    if len(self.all_loss_log) < 2:
      return torch.tensor(1.).cuda()
    else:
      tmp = 0
      for k, _ in self.task_priority.items():
         tmp += self.task_priority[k].cuda() * diff_weights[k]
      return (1/tmp).cuda()