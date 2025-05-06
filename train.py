import click
from diffusions import DDPM, DDPMConfig, DDIMConfig, DDIM, Trainer
import os
from pathlib import Path
import wandb
import yaml

@click.command()
@click.option("--config_path", type=Path, required=True)
@click.option("--model", type=str, required=True)
@click.option("--wandb_key", type=str, required=False)
def main(config_path: Path, model: str, wandb_key: str = ""):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if model == "ddpm":
        config = DDPMConfig(**data)
        diffusion_model = DDPM(config)
    elif model == "ddim":
        config = DDIMConfig(**data)
        diffusion_model = DDIM(config)
    else:
        raise AttributeError("Unknown model type")

    if config.use_wandb:
        if not wandb_key:
            raise RuntimeError("Wandb usage is turned on but wandb key wasn't passed.")

        os.environ['WANDB_API_KEY'] = wandb_key
        wandb.login()

        wandb.init(
            project="ddpm_project",
            name="ddpm_run")

    trainer = Trainer(diffusion_model, config)
    trainer.train()

if __name__ == "__main__":
    main()


