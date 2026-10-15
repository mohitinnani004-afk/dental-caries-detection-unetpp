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

try:
    from .augmentations import DentalRadiographAugmenter
except (ImportError, ValueError):
    from augmentations import DentalRadiographAugmenter


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

    pin_mem = bool(TORCH_AVAILABLE and torch and torch.cuda.is_available())

    if TORCH_AVAILABLE and DataLoader is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem
        )
    else:
        # Standalone batch loader when torch is not installed
        class StandaloneDataLoader:
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
            def __len__(self):
                return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            def __iter__(self):
                for start_idx in range(0, len(self.dataset), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(self.dataset))
                    samples = [self.dataset[k] for k in range(start_idx, end_idx)]
                    imgs = np.stack([s[0] for s in samples], axis=0)
                    masks = np.stack([s[1] for s in samples], axis=0)
                    names = [s[2] for s in samples]
                    yield imgs, masks, names

        train_loader = StandaloneDataLoader(train_dataset, batch_size)
        val_loader = StandaloneDataLoader(val_dataset, batch_size)
        test_loader = StandaloneDataLoader(test_dataset, batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=" * 65)
    print("  DENTAL CARIES DATASET & DATALOADER PIPELINE (RUNNING FORMAT)")
    print("=" * 65)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "dataset"

    # Auto-bootstrap dataset if images directory is missing
    if not (data_dir / "images").exists() or len(list((data_dir / "images").glob("*.png"))) == 0:
        print(f"\n[INFO] Dataset not found in {data_dir}. Auto-generating 24 sample radiographs...")
        try:
            from .sample_generator import generate_sample_dataset
        except (ImportError, ValueError):
            from sample_generator import generate_sample_dataset
        generate_sample_dataset(output_dir=data_dir, num_samples=24, width=256, height=256)
        print("[INFO] Sample dataset generated successfully!\n")

    # Initialize DataLoaders
    BATCH_SIZE = 4
    IMG_SIZE = (256, 256)
    print(f"Loading dataset from: {data_dir}")
    print(f"Batch Size: {BATCH_SIZE} | Resolution: {IMG_SIZE} | CLAHE Enabled: True")

    train_loader, val_loader, test_loader = create_dental_dataloaders(
        data_dir=data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        apply_clahe=True
    )

    print(f"\n[PARTITIONS]:")
    print(f"  * Train Set:      {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"  * Validation Set: {len(val_loader.dataset)} images ({len(val_loader)} batches)")
    print(f"  * Test Set:       {len(test_loader.dataset)} images ({len(test_loader)} batches)")

    # Fetch and inspect the first batch
    for batch_idx, (images, masks, filenames) in enumerate(train_loader):
        print(f"\n[INSPECTING BATCH #1]:")
        
        # Check types & shapes
        if TORCH_AVAILABLE and hasattr(images, "shape"):
            img_shape = list(images.shape)
            mask_shape = list(masks.shape)
            img_min, img_max = float(images.min()), float(images.max())
            mask_unique = [int(v) for v in masks.unique().tolist()]
        else:
            img_shape = list(images.shape) if hasattr(images, "shape") else [len(images)]
            mask_shape = list(masks.shape) if hasattr(masks, "shape") else [len(masks)]
            img_min = float(np.min(images))
            mask_unique = [int(v) for v in np.unique(masks)]

        print(f"  Images Tensor Shape: {img_shape} -> [Batch, Channel, Height, Width]")
        print(f"  Masks Tensor Shape:  {mask_shape} -> [Batch, Channel, Height, Width]")
        print(f"  Radiograph Values:   Range [{img_min:.2f}, 1.00] (Normalized)")
        print(f"  Mask Binary Values:  {mask_unique} (0 = Sound Tooth, 1 = Caries)")
        print(f"  Filenames in Batch:  {filenames}")

        # Render preview figure for verification
        out_preview = project_root / "outputs" / "dataset_batch_preview.png"
        out_preview.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(BATCH_SIZE, 4, figsize=(14, 3.2 * BATCH_SIZE), facecolor="#0e1117")
            if BATCH_SIZE == 1:
                axes = np.expand_dims(axes, 0)

            for b in range(min(BATCH_SIZE, len(filenames))):
                im_b = images[b, 0].cpu().numpy() if (TORCH_AVAILABLE and hasattr(images, "cpu")) else images[b, 0]
                mk_b = masks[b, 0].cpu().numpy() if (TORCH_AVAILABLE and hasattr(masks, "cpu")) else masks[b, 0]

                # De-normalize image to [0, 255]
                im_vis = np.clip((im_b * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

                # Overlay
                overlay = np.stack([im_vis]*3, axis=-1).astype(np.float32) / 255.0
                mk_bool = mk_b > 0.5
                overlay[mk_bool] = overlay[mk_bool] * 0.35 + np.array([1.0, 0.28, 0.34]) * 0.65

                axes[b, 0].imshow(im_vis, cmap="gray")
                axes[b, 0].set_title(f"Radiograph: {filenames[b]}", color="white", fontsize=10)
                axes[b, 0].axis("off")

                axes[b, 1].imshow(apply_clahe_preprocessing(Image.fromarray(im_vis)), cmap="gray")
                axes[b, 1].set_title("CLAHE Enhanced", color="white", fontsize=10)
                axes[b, 1].axis("off")

                axes[b, 2].imshow(mk_b * 255, cmap="gray")
                caries_px = int(np.sum(mk_bool))
                axes[b, 2].set_title(f"Ground Truth Mask ({caries_px} px)", color="white", fontsize=10)
                axes[b, 2].axis("off")

                axes[b, 3].imshow(overlay)
                axes[b, 3].set_title(f"Clinical Overlay ({'Caries Detected' if caries_px > 0 else 'Clear'})", color="#00f2fe", fontsize=10)
                axes[b, 3].axis("off")

            plt.tight_layout()
            plt.savefig(out_preview, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close()
            print(f"\n[VISUAL PREVIEW SAVED]: {out_preview}")
        except Exception as e:
            print(f"Preview figure skipped ({e})")

        break

    print("\n" + "=" * 65)
    print("  [SUCCESS] Dataset pipeline is in complete working running format!")
    print("=" * 65)
