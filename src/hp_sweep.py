import hydra
from utils.utils_tasks import get_config
from run_glue_method_hp import train_eval_glue_model
from omegaconf import OmegaConf
import wandb

def train(config):
    config = OmegaConf.create(config)
    train_eval_glue_model(config, config.training, config.data)


@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    dict_config = OmegaConf.to_container(config, resolve=True)
    sweep_id = wandb.sweep(dict_config, project="sample")
    wandb.agent(sweep_id, train, count=5)



if __name__ == "__main__":
    main()