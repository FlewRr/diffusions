import click
from diffusions import DDPM, DDPMConfig, Trainer
import os
from pathlib import Path
import wandb
import yaml

@click.command()
@click.option("--config_path", type=Path, required=True)
@click.option("--wandb_key", type=str, required=False)
def main(config_path: Path, wandb_key: str = ""):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    ddpm_config = DDPMConfig(**data)
    ddpm = DDPM(ddpm_config)

    if ddpm_config.use_wandb and wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        wandb.login()

        wandb.init(
            project="ddpm_project",
            name="ddpm_run")

    trainer = Trainer(ddpm, ddpm_config)
    trainer.train()

if __name__ == "__main__":
    main()
# TODO: create trainer config where model info is created (maybe)











# from diffusions import DDIM, DDIMConfig, Trainer
# import yaml
#
# path = "C:/Users/ПК/Desktop/diffusions/configs/ddim_config.yaml"
# with open(path, "r") as f:
#     data = yaml.safe_load(f)
#
# ddim_config = DDIMConfig(**data)
# ddim = DDIM(ddim_config)
#
# trainer = Trainer(ddim, ddim_config)
# trainer.train()