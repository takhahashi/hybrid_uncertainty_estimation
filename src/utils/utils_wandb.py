import logging
import os

import wandb

log = logging.getLogger(__name__)


def _wandb_log(_dict):
    if wandb.run is not None:
        wandb.log(_dict)
    else:
        log.info(repr(_dict))


wandb.log_opt = _wandb_log


def init_wandb(directory, config, wandb_disabled=False):
    if "NO_WANDB" in os.environ and os.environ["NO_WANDB"] == "true":
        ## working without wandb :c
        log.info("== Working without wandb")
        return None

    # setting up env variables
    # os.environ["WANDB_ENTITY"] = "artem105"
    os.environ["WANDB_PROJECT"] = "uncertainty-estimation"
    if wandb_disabled:
        os.environ["WANDB_DISABLED"] = "true"

    # generating group name and run name
    directory_contents = directory.split("/")
    run_name = directory_contents[-1]  # ${now:%H-%M-%S}-${repeat}
    date = directory_contents[-2]  # ${now:%Y-%m-%d}
    strat_name = directory_contents[-3]  # ${al.strat_name}
    model_name = directory_contents[
        -4
    ]  # ${model.model_type}_${model.classifier} for BERT
    task = directory_contents[-5]  # ${data.task}

    group_name = f"{task}|{model_name}|{strat_name}|{date}"
    run_name = f"{run_name}"
    print('config:',config)
    print()
    if 'deberta' in config.model.model_name_or_path:
        model_name = 'deberta'
    else:
        model_name = 'bert'

    if "hybrid" in config.model.model_name_or_path:
        inf_type = 'hybrid'
    elif "is_regression" in config.model.keys() and config.model.is_regression:
        inf_type = 'regression'
    else:
        inf_type = 'classification'

    if wandb_disabled:
        project_name=None
        run_name=None
    else:
        if config.data.task_name == 'asap':
            project_name = f'{model_name}_' + f'{inf_type}_' + f'{config.ue.reg_type}_' + f'PromptId-{config.data.prompt_id}'
        elif config.data.task_name == 'riken':
            project_name = f'{model_name}_' + f'{inf_type}_' + f'{config.ue.reg_type}_' + f'{config.data.question_id}_{config.data.prompt_id}_{config.data.score_id}'
        run_name = f'fold_{config.data.fold}'
    return wandb.init(
        project = project_name,
        name=run_name,
    )
