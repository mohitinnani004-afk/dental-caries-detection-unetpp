"""
Configuration settings for Dental Caries Detection using UNet++
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = PROJECT_ROOT / "samples"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Ensure directories exist
for p in [CHECKPOINTS_DIR, OUTPUTS_DIR, SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """UNet++ Model Architecture Configuration."""
    name: str = "UNetPlusPlus"
    in_channels: int = 1         # 1 for grayscale dental X-rays (or 3 for RGB)
    num_classes: int = 1         # Binary segmentation: Caries vs Background
    deep_supervision: bool = True
    # Filter depths for nested levels X^{0,j} to X^{4,j}
    nb_filter: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    use_attention: bool = True   # Spatial attention gate on skip connections
    dropout_rate: float = 0.15
    norm_type: str = "batchnorm" # "batchnorm" or "instancenorm"


@dataclass
class DataConfig:
    """Dental Radiograph Dataset Configuration."""
    image_size: Tuple[int, int] = (256, 256) # (Height, Width)
    apply_clahe: bool = True     # Contrast Limited Adaptive Histogram Equalization
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training Hyperparameters Configuration."""
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = "adamw"      # "adam", "adamw", or "sgd"
    scheduler: str = "cosine"     # "cosine", "plateau", or "step"
    early_stopping_patience: int = 12
    mixed_precision: bool = True  # AMP (torch.cuda.amp)
    num_workers: int = 2
    
    # Loss weights
    bce_weight: float = 0.5       # Weight for BCE component in BCEDiceLoss
    dice_weight: float = 0.5      # Weight for Dice component in BCEDiceLoss
    # Weights for deep supervision heads [L1, L2, L3, L4]
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4]
    )


@dataclass
class InferenceConfig:
    """Inference & Clinical Lesion Detection Configuration."""
    probability_threshold: float = 0.50
    min_lesion_area_pixels: int = 15      # Ignore micro noise artifacts
    connectivity: int = 8                # 4- or 8-way pixel connectivity
    # Lesion Severity depth thresholds (% penetration relative to tooth boundary)
    enamel_threshold: float = 0.30       # E1/E2 superficial to deep enamel
    dentin_threshold: float = 0.70       # D1/D2 outer to inner dentin
    pulp_risk_threshold: float = 0.85    # D3 / Pulpal involvement risk
    test_time_augmentation: bool = True  # Horizontal flip TTA


@dataclass
class AppConfig:
    """Interactive Diagnostic Dashboard Configuration."""
    host: str = "127.0.0.1"
    port: int = 8501
    debug: bool = False
    upload_folder: Path = PROJECT_ROOT / "web_app" / "uploads"
    allowed_extensions: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


# Default instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_TRAIN_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
DEFAULT_APP_CONFIG = AppConfig()
