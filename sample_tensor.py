import click
from diffusions import DDPM, DDPMConfig, DDIMConfig, DDIM, Trainer
import os
from pathlib import Path
import torch
import wandb
import yaml

@click.command()
@click.option("--config_path", type=Path, required=True)
@click.option("--model", type=str, required=True)
@click.option("--num_samples", type=int, required=True)
@click.option("--path_to_save", type=str, required=False)
def main(config_path: Path, model: str, num_samples: int, path_to_save: str=""):
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

    try:
        checkpoint_path = config.checkpoint_path
        diffusion_model.load_state_dict(checkpoint_path)
    except:
        print("Something went wrong while loading weights, model is set to default weights.")

    BATCH_SIZE = 100
    TOTAL_SAMPLES = 50000

    all_samples = []

    for _ in range(TOTAL_SAMPLES // BATCH_SIZE):
        sampled_images = diffusion_model.sample_images(num_samples=BATCH_SIZE)
        all_samples.append(sampled_images)

    all_samples = torch.cat(all_samples, dim=0)


    if not os.path.exists(path_to_save):
        print("No such path")
        return

    torch.save(all_samples, path_to_save + "samples.pt")

if __name__ == "__main__":
    main()


