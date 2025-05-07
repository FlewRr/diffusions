from diffusions.config import DatasetConfig, SchedulerConfig
W, Field
from typing import Tuple, Annotated, Literal, Optional

class DDPMConfig(BaseModel):
    batch_size: int
    epochs: int

    input_channels: int
    output_channels: int
    hidden_channels: int
    time_embedding_dim: int
    image_size: Tuple[int, int]
    image_channels: int

    lr: float
    betas: Tuple[float, float]
    device: str

    use_wandb: bool
    use_ema: bool
    use_amp: bool
    ema_decay: float
    eval_num_samples: int
    eval_sampling_epochs: int
    save_checkpoints: bool
    save_checkpoints_epochs: int
    save_checkpoints_path: str
    num_workers: int
    max_grad_norm: float

    checkpoint_path: Optional[str] = ""

    scheduler: SchedulerConfig
    dataset: DatasetConfig
