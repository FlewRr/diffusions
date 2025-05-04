## Diffusions

This repository contains PyTorch implementation of DDPM and DDIM from original papers including training and sampling code.

# Installation
1. Clone the repository
```
git clone https://github.com/FlewRr/diffusions.git
cd diffusions
```
2. Create and activate a Python virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install the required dependencies
```
pip install -r requirements.txt
```

# Training
To train the model run the following script in the root directory
```
python train.py --config_path {your_config_path} --model {model}
```
Sampling Parameters:
* config_path: Path to yaml config
* model: Either ddpm or ddim
* num_samples: Number of samples to generate.
* path_to_save: Path to directory where images will be stored

# Sampling from Trained Models
To sample use this code in the root directory
```
python sample.py --config_path={your_config_path} --model={model} --checkpoint_path={path_to_checkpoint} --num_samples={number of samples} --path_to_save={dir_to_save}
```

Sampling Parameters:
* config_path: Path to yaml config
* model: Either ddpm or ddim
* checkpoint_path: Path to the checkpoint of the unet (may be empty)
* num_samples: Number of samples to generate.
* path_to_save: Path to directory where images will be stored


# Configuration
Yaml configs are used as the configuration for models and training.

Configurable parameters:
* batch_size, epochs, lr: Training configuration
* input_channels, output_channels, hidden_channels, time_embedding_dim, image_size, image_channels: U-Net channels configuration
* data_path: Dataset configuration (if use_cifar10 is True, will use cifar10 for training)
* scheduler: Scheduling type (cosine, linear), timesteps, s (for cosine), beta_min/beta_max (for linear)
* use_wandb, use_amp, use_ema, ema_decay: Wandb, amp, EMA usage in training loop
* eval_num_samples, eval_sampling_epochs, save_checkpoints, save_checkpoints_path, save_checkpoints_epochs: Saving and evaluation configuration
* ddim_timesteps, eta: Parameters for DDIM sampling
