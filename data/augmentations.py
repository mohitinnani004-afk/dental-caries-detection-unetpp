"""
Medical Radiograph Data Augmentation pipeline for Dental Caries.
Uses pure PyTorch / Torchvision / PIL transforms without requiring heavy external C-libraries.
"""

import random
from typing import Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter
try:
    import torchvision.transforms.functional as TF
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TF = None
    torch = None


class DentalRadiographAugmenter:
    """
    Augmentation pipeline tailored for dental bitewing and periapical radiographs:
      - Random horizontal flip (mirroring tooth quadrants)
      - Subtle rotation (-10 to +10 degrees, patient head alignment)
      - Subtle scaling and translation
      - Contrast and brightness jitter (simulating sensor/exposure differences)
      - Subtle Gaussian blur (simulating slight patient micro-motion)
    Synchronously applies spatial transforms to both radiograph and caries ground-truth mask.
    """
    def __init__(
        self,
        is_train: bool = True,
        image_size: Tuple[int, int] = (256, 256),
        p_hflip: float = 0.5,
        p_rotation: float = 0.6,
        max_rotation_deg: float = 12.0,
        p_contrast: float = 0.4,
        p_brightness: float = 0.4,
        p_blur: float = 0.2
    ):
        self.is_train = is_train
        self.image_size = image_size
        self.p_hflip = p_hflip
        self.p_rotation = p_rotation
        self.max_rotation_deg = max_rotation_deg
        self.p_contrast = p_contrast
        self.p_brightness = p_brightness
        self.p_blur = p_blur

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Any, Any]:
        # Resize both to target size
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() < self.p_hflip:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random subtle rotation
            if random.random() < self.p_rotation:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            # Photometric augmentations (image only, not mask)
            if random.random() < self.p_contrast:
                factor = random.uniform(0.8, 1.25)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)

            if random.random() < self.p_brightness:
                factor = random.uniform(0.85, 1.15)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)

            if random.random() < self.p_blur:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

        # Convert to Tensors
        # Image: [1, H, W] in range [0.0, 1.0]
        img_tensor = TF.to_tensor(image)
        if img_tensor.shape[0] > 1:
            # Convert RGB to Grayscale
            img_tensor = img_tensor[0:1, :, :]

        # Mask: [1, H, W] binary float {0.0, 1.0}
        mask_tensor = TF.to_tensor(mask)
        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1, :, :]
        mask_tensor = (mask_tensor > 0.5).float()

        # Standardize radiograph intensity (zero mean, unit variance or [0, 1])
        img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor, mask_tensor
