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
