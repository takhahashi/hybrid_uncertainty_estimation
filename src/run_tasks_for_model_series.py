import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import hydra
import yaml
from pathlib import Path
from collections.abc import Iterable

import utils.utils_tasks as utils

import logging
import pdb

log = logging.getLogger(__name__)


def run_tasks(config_path, cuda_devices):
    cuda_devices = (
        cuda_devices if isinstance(cuda_devices, Iterable) else [cuda_devices]
    )
    cuda_devices_str = "[" + ",".join(str(e) for e in cuda_devices) + "]"
    command = f"HYDRA_CONFIG_PATH={config_path} python run_tasks_on_multiple_gpus.py cuda_devices={cuda_devices_str}"
    log.info(f"Command: {command}")
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError(ret)

    return ret


def run_glue_for_model_series_fast(config, work_dir):
    tasks = []
    path_exists = os.path.exists(config.model_series_dir)
    if "fairlib" in str(config.model_series_dir) and "ensemble" not in str(
        config.model_series_dir
    ):
        listdir = [f"{config.model_series_dir}_{seed}" for seed in config.seeds]
    elif path_exists:
        listdir = os.listdir(config.model_series_dir)
    else:
        exit()
        print(config.seeds)
        listdir = [int(seed) for seed in config.seeds]
    for model_dir_name in listdir:
        if "fairlib" in str(config.model_series_dir) and "ensemble" not in str(
            config.model_series_dir
        ):
            model_path = Path(model_dir_name) / "models"
        elif path_exists:
            model_path = Path(config.model_series_dir) / model_dir_name
            if "fairlib" in str(config.model_series_dir):
                # ensemble & fairlib case
                model_path = Path(model_path) / "models"
            if config.script == 'run_gp.py':
                model_path = Path(model_path) / "model"
        else:
            model_path = config.model_series_dir

        model_args_str = config.args
        model_args_str += " "
        model_args_str += f"model.model_name_or_path={model_path}"

        if "fairlib" in str(config.model_series_dir) and "ensemble" not in str(
            config.model_series_dir
        ):
            seed = model_dir_name.split("_")[-1]
        elif "ensemble" in str(config.output_dir):
            seed = config.output_dir.split("/")[-1]
        else:
            seed = str(model_dir_name)

        args_str = model_args_str
        args_str += " "
        if "fairlib" in str(config.model_series_dir) and "ensemble" not in str(
            config.model_series_dir
        ):
            output_dir = str(Path(work_dir) / "results" / str(seed))
        else:
            output_dir = str(Path(work_dir) / "results" / str(model_dir_name))
        args_str += f"hydra.run.dir={output_dir}"
        args_str += " "
        args_str += f"output_dir={output_dir}"
        args_str += " "
        args_str += "do_train=False do_eval=True"
        if 'asap' in config.config_path:
            model_type = config.model_series_dir.split('/')[-3].split('_')[0]
            args_str += " "
            args_str += f"data.prompt_id={config.model_series_dir[-1]}"
            args_str += " "
            args_str += f"data.fold={model_dir_name[-1]}"
            args_str += " "
            args_str += f"model.model_type={model_type}"
        elif 'riken' in config.config_path:
            print(model_dir_name)
            model_type = config.model_series_dir.split('/')[-3].split('_')[0]
            riken_ids = config.model_series_dir.split('/')[-1].split('_')
            question_id = riken_ids[0]
            prompt_id = riken_ids[1]+'_'+riken_ids[2]+'_'+riken_ids[3]
            score_id = '_'.join(riken_ids[4:])
            args_str += " "
            args_str += f"data.question_id={question_id}"
            args_str += " "
            args_str += f"data.prompt_id={prompt_id}"
            args_str += " "
            args_str += f"data.score_id={score_id}"
            args_str += " "
            args_str += f"data.fold={model_dir_name[-1]}"
            args_str += " "
            args_str += f"model.model_type={model_type}"
        else:
            raise ValueError(f"PATH:{config.config_path} is INVALID!")


        if config.script == 'run_gp.py':
            encoder_model_path = '/'.join(str(model_path).replace('GP', 'classification').split('/')[:-1])
            args_str += " "
            args_str += f"++encoder_model.model_name_or_path={encoder_model_path}"
            args_str += " "
            args_str += f"++encoder_model.model_type=classification"


        task = {
            "config_path": config.config_path,
            "environ": "",
            "command": config.script,
            "name": f"model_{model_dir_name}"
            if "fairlib" not in str(config.model_series_dir)
            else f"model_fairlib_{seed}",
            "args": args_str,
        }

        tasks.append(task)

    config_path = Path(work_dir) / "config.yaml"
    config_structure = {}
    config_structure["cuda_devices"] = ""
    config_structure["tasks"] = tasks
    config_structure["hydra"] = {"run": {"dir": work_dir}}
    with open(config_path, "w") as f:
        yaml.dump(config_structure, f)

    run_tasks(config_path, config.cuda_devices)


if __name__ == "__main__":

    @hydra.main(
        config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
        config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
    )
    def main(config):
        auto_generated_dir = os.getcwd()
        log.info(f"Work dir: {auto_generated_dir}")
        os.chdir(hydra.utils.get_original_cwd())

        run_glue_for_model_series_fast(config, auto_generated_dir)

    main()
