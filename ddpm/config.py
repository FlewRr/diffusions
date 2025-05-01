from pydantic import BaseModel, Field
from typing import Tuple, Annotated, Literal, Optional


class BaseSchedulerConfig(BaseModel):
    timesteps: int


class LinearSchedulerConfig(BaseSchedulerConfig):
    scheduler: Literal["linear"] = "linear"
    beta_min: float
    beta_max: float


class CosineSchedulerConfig(BaseSchedulerConfig):
    scheduler: Literal["cosine"] = "cosine"
    s: float


SchedulerConfig = Annotated[LinearSchedulerConfig | CosineSchedulerConfig,  Field(discriminator="scheduler")]

class DatasetConfig(BaseModel):
    data_path: str
    use_cifar10: Optional[bool] = False

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
    ema_decay: float
    eval_num_samples: int
    eval_sampling_epochs: int
    save_checkpoints: bool
    save_checkpoints_epochs: int
    save_checkpoints_path: str

    scheduler: SchedulerConfig
    dataset: DatasetConfig
