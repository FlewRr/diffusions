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

