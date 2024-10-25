from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    content_image_URL: str
    style_image_URL: str
    

@dataclass(frozen=True)
class PretrainModelConfig:
    root_dir: Path
    weights_dir: Path

@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    learning_rate: float 
    beta_1: float 
    epsilon: float 
    total_variation_weight: int 
    epochs: int 
    steps_per_epoch: int 
    style_weight: float 
    content_weight: float 
    max_dim: int
    style_layers: list
    content_layers: list


@dataclass(frozen=True)
class EvaluateConfig:
    root_dir: Path
    saved_model_dir: Path
    keras_saved_model_dir: Path
    tf_saved_model_dir: Path
    save_keras: bool
    image_saved_dir: Path
