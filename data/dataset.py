"""
PyTorch Dataset for Dental Caries Detection and Segmentation.
Handles radiograph loading, CLAHE preprocessing, and train/val/test splits.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None
    class Dataset:
        pass

from .augmentations import DentalRadiographAugmenter


def apply_clahe_preprocessing(
    image: Image.Image,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8)
) -> Image.Image:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Significantly enhances radiolucent caries lesions in low-contrast dental X-rays.
    """
    img_np = np.array(image.convert("L"))

    # Try cv2 CLAHE if available, otherwise fallback to local rank histogram
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced)
    except ImportError:
        # Fallback using PIL ImageOps or custom tile equalization
        from PIL import ImageOps
        return ImageOps.autocontrast(image.convert("L"), cutoff=2)


class DentalCariesDataset(Dataset):
    """
    PyTorch Dataset for Dental Caries Radiographs and Ground-Truth Masks.
    
    Expected Directory Structure:
      data_dir/
        images/
          001.png, 002.png, ...
        masks/
          001.png, 002.png, ...
    """
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        is_train: bool = True,
        image_size: Tuple[int, int] = (256, 256),
        apply_clahe: bool = True,
        transform: Optional[Callable] = None
    ):
        assert len(image_paths) == len(mask_paths), (
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.transform = transform or DentalRadiographAugmenter(
            is_train=is_train,
            image_size=image_size
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, str]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image & mask as grayscale
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.apply_clahe:
            image = apply_clahe_preprocessing(image)

        # Apply augmentation & tensor formatting
        img_tensor, mask_tensor = self.transform(image, mask)

        return img_tensor, mask_tensor, img_path.name


def create_dental_dataloaders(
    data_dir: Union[str, Path],
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    apply_clahe: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Scans data directory, splits dataset into train/val/test partitions,
    and returns corresponding PyTorch DataLoaders.
    """
    data_path = Path(data_dir)
    img_dir = data_path / "images"
    mask_dir = data_path / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images' and 'masks' subdirectories under {data_dir}"
        )

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    all_images = []
    for ext in extensions:
        all_images.extend(sorted(img_dir.glob(ext)))

    if not all_images:
        raise ValueError(f"No image files found in {img_dir}")

    valid_images = []
    valid_masks = []
    for img_p in all_images:
        # Match mask with same stem
        mask_matches = list(mask_dir.glob(f"{img_p.stem}.*"))
        if mask_matches:
            valid_images.append(img_p)
            valid_masks.append(mask_matches[0])

    if not valid_images:
        raise ValueError(f"No matching masks found in {mask_dir}")

    total_samples = len(valid_images)
    indices = np.random.RandomState(seed).permutation(total_samples)

    n_train = int(total_samples * train_split)
    n_val = int(total_samples * val_split)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_dataset = DentalCariesDataset(
        image_paths=[valid_images[i] for i in train_idx],
        mask_paths=[valid_masks[i] for i in train_idx],
        is_train=True,
        image_size=image_size,
        apply_clahe=apply_clahe
    )

    val_dataset = DentalCariesDataset(
        image_paths=[valid_images[i] for i in val_idx],
        mask_paths=[valid_masks[i] for i in val_idx],
        is_train=False,
        image_size=image_size,
        apply_clahe=apply_clahe
    )

    test_dataset = DentalCariesDataset(
        image_paths=[valid_images[i] for i in test_idx],
        mask_paths=[valid_masks[i] for i in test_idx],
        is_train=False,
        image_size=image_size,
        apply_clahe=apply_clahe
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
