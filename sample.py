import click
from diffusions import DDPM, DDPMConfig, DDIMConfig, DDIM, Trainer
import os
from pathlib import Path
import wandb
import yaml

@click.command()
@click.option("--config_path", type=Path, required=True)
@click.option("--model", type=str, required=True)
@click.option("--checkpoint_path", type=str, required=True)
@click.option("--num_samples", type=int, required=True)
@click.option("--path_to_save", type=str, required=False)
def main(config_path: Path, model: str, checkpoint_path: str, num_samples: int, path_to_save: str=""):
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
        diffusion_model.load_from_checkpoint(checkpoint_path)
    except:
        print("Something went wrong while loading weights, model is set to default weights.")

    sampled_images = diffusion_model.sample_images(num_samples=num_samples)

    if not os.path.exists(path_to_save):
        if not os.path.exists("images/"):
            os.system("mkdir images")

        path_to_save = "images"

    for i, img in enumerate(sampled_images):
        img.save(path_to_save + f"/{i}.png")


if __name__ == "__main__":
    main()


