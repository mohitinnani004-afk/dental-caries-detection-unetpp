"""
Synthetic Dental Radiograph and Caries Mask Generator.
Allows instant testing, training, validation, and interactive UI demonstration
without requiring pre-download of private clinical radiology datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_single_radiograph(
    width: int = 256,
    height: int = 256,
    has_caries: bool = True,
    lesion_type: str = "proximal",  # "proximal", "occlusal", "cervical"
    noise_level: float = 0.08,
    seed: Optional[int] = None
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    Generates a realistic synthetic bitewing/periapical dental radiograph and ground-truth caries mask.
    Simulates anatomical tooth structures (enamel, dentin, pulp) and bone trabeculae.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Base background: Alveolar bone with trabecular texture (intensity ~ 60-90)
    base_arr = np.random.normal(loc=75, scale=12, size=(height, width)).astype(np.float32)

    # Add low-frequency bone density variation
    y_coords, x_coords = np.ogrid[:height, :width]
    bone_gradient = 15.0 * np.sin(x_coords / 40.0) + 10.0 * np.cos(y_coords / 50.0)
    base_arr += bone_gradient
    base_arr = np.clip(base_arr, 20, 110)

    # Convert to grayscale image for drawing anatomical tooth crowns and roots
    img = Image.fromarray(base_arr.astype(np.uint8), mode="L")
    mask = Image.new("L", (width, height), 0)

    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    # Create 2 to 3 adjacent teeth (e.g. Molar and Premolar)
    tooth_centers = [int(width * 0.28), int(width * 0.72)]
    crown_top = int(height * 0.18)
    crown_bottom = int(height * 0.60)
    root_bottom = int(height * 0.92)

    tooth_boxes = []

    for c_x in tooth_centers:
        w_t = int(width * 0.38)
        left = c_x - w_t // 2
        right = c_x + w_t // 2
        tooth_boxes.append((left, crown_top, right, root_bottom))

        # 1. Outer Enamel (Radiopaque - high attenuation, bright: ~210-240)
        enamel_coords = [
            (c_x - w_t * 0.45, crown_bottom),
            (c_x - w_t * 0.50, crown_top + 30),
            (c_x - w_t * 0.40, crown_top + 8),
            (c_x - w_t * 0.20, crown_top),
            (c_x, crown_top + 10),  # Central occlusal groove
            (c_x + w_t * 0.20, crown_top),
            (c_x + w_t * 0.40, crown_top + 8),
            (c_x + w_t * 0.50, crown_top + 30),
            (c_x + w_t * 0.45, crown_bottom),
            # Root outline
            (c_x + w_t * 0.30, root_bottom - 20),
            (c_x + w_t * 0.15, root_bottom),
            (c_x, root_bottom - 40), # Bifurcation
            (c_x - w_t * 0.15, root_bottom),
            (c_x - w_t * 0.30, root_bottom - 20),
        ]
        draw_img.polygon(enamel_coords, fill=225)

        # 2. Dentin Core (Intermediate radiopacity: ~160-185)
        # Scaled inward by 10-14 pixels
        dentin_coords = [
            (c_x - w_t * 0.36, crown_bottom - 5),
            (c_x - w_t * 0.38, crown_top + 35),
            (c_x - w_t * 0.30, crown_top + 20),
            (c_x - w_t * 0.15, crown_top + 16),
            (c_x, crown_top + 22),
            (c_x + w_t * 0.15, crown_top + 16),
            (c_x + w_t * 0.30, crown_top + 20),
            (c_x + w_t * 0.38, crown_top + 35),
            (c_x + w_t * 0.36, crown_bottom - 5),
            (c_x + w_t * 0.24, root_bottom - 25),
            (c_x + w_t * 0.10, root_bottom - 10),
            (c_x, root_bottom - 45),
            (c_x - w_t * 0.10, root_bottom - 10),
            (c_x - w_t * 0.24, root_bottom - 25),
        ]
        draw_img.polygon(dentin_coords, fill=165)

        # 3. Pulp Chamber and Root Canals (Radiolucent soft tissue: dark ~40-60)
        pulp_coords = [
            (c_x - w_t * 0.15, crown_top + 45),
            (c_x - w_t * 0.08, crown_top + 38),  # Pulp horn
            (c_x, crown_top + 48),
            (c_x + w_t * 0.08, crown_top + 38),  # Pulp horn
            (c_x + w_t * 0.15, crown_top + 45),
            (c_x + w_t * 0.12, crown_bottom),
            (c_x + w_t * 0.05, root_bottom - 30), # Root canal
            (c_x + w_t * 0.02, root_bottom - 15),
            (c_x - w_t * 0.02, root_bottom - 15),
            (c_x - w_t * 0.05, root_bottom - 30),
            (c_x - w_t * 0.12, crown_bottom),
        ]
        draw_img.polygon(pulp_coords, fill=45)

    lesion_meta = {
        "has_caries": has_caries,
        "lesions": []
    }

    if has_caries:
        # Generate 1 to 2 caries lesions
        num_lesions = random.choice([1, 1, 2])
        for l_i in range(num_lesions):
            t_choice = random.choice([0, 1])
            c_x = tooth_centers[t_choice]
            t_w = int(width * 0.38)

            chosen_type = lesion_type if l_i == 0 else random.choice(["proximal", "occlusal"])

            if chosen_type == "proximal":
                # Interproximal contact point caries (radiolucent demineralization)
                # Outer enamel toward inner dentin
                side = 1 if (c_x < width / 2) else -1
                caries_x = int(c_x + side * (t_w * 0.42))
                caries_y = int(crown_top + random.uniform(22, 50))
                rx = random.randint(7, 14)
                ry = random.randint(10, 18)

                # Draw radiolucent lesion (dark demineralized area: ~40-70 intensity)
                draw_img.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=random.randint(48, 75)
                )
                draw_mask.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=255
                )

                lesion_meta["lesions"].append({
                    "id": l_i + 1,
                    "type": "proximal",
                    "bbox": [caries_x - rx, caries_y - ry, rx * 2, ry * 2],
                    "center": [caries_x, caries_y],
                    "severity": "Dentin (D1/D2)" if rx > 9 else "Enamel (E2)"
                })

            elif chosen_type == "occlusal":
                # Occlusal fissure caries at central groove
                caries_x = int(c_x + random.uniform(-6, 6))
                caries_y = int(crown_top + 16)
                rx = random.randint(8, 15)
                ry = random.randint(6, 12)

                draw_img.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=random.randint(52, 78)
                )
                draw_mask.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=255
                )

                lesion_meta["lesions"].append({
                    "id": l_i + 1,
                    "type": "occlusal",
                    "bbox": [caries_x - rx, caries_y - ry, rx * 2, ry * 2],
                    "center": [caries_x, caries_y],
                    "severity": "Enamel/Dentin (E2/D1)"
                })

            elif chosen_type == "cervical":
                # Cervical / cementoenamel root caries
                side = random.choice([-1, 1])
                caries_x = int(c_x + side * (t_w * 0.38))
                caries_y = int(crown_bottom + random.uniform(-5, 12))
                rx = random.randint(8, 12)
                ry = random.randint(7, 13)

                draw_img.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=random.randint(50, 72)
                )
                draw_mask.ellipse(
                    [caries_x - rx, caries_y - ry, caries_x + rx, caries_y + ry],
                    fill=255
                )

                lesion_meta["lesions"].append({
                    "id": l_i + 1,
                    "type": "cervical",
                    "bbox": [caries_x - rx, caries_y - ry, rx * 2, ry * 2],
                    "center": [caries_x, caries_y],
                    "severity": "Cervical / Root Caries"
                })

    # Apply realistic radiographic blur (penumbra effect in dental X-rays)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Add sensor noise (simulating quantum mottle in digital radiography sensors)
    final_np = np.array(img, dtype=np.float32)
    sensor_noise = np.random.normal(loc=0.0, scale=noise_level * 255.0, size=(height, width))
    final_np = np.clip(final_np + sensor_noise, 0, 255).astype(np.uint8)

    final_img = Image.fromarray(final_np, mode="L")
    return final_img, mask, lesion_meta


def generate_sample_dataset(
    output_dir: Path,
    num_samples: int = 40,
    caries_ratio: float = 0.85,
    width: int = 256,
    height: int = 256
) -> Tuple[List[Path], List[Path]]:
    """
    Generates a directory of synthetic radiographs and masks for training/validation.
    """
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    mask_paths = []

    types = ["proximal", "occlusal", "cervical"]

    for i in range(num_samples):
        has_caries = (i < int(num_samples * caries_ratio))
        l_type = types[i % len(types)]
        img, mask, _ = generate_single_radiograph(
            width=width,
            height=height,
            has_caries=has_caries,
            lesion_type=l_type,
            seed=1000 + i
        )

        name = f"dental_xray_{i+1:04d}.png"
        img_p = img_dir / name
        mask_p = mask_dir / name

        img.save(img_p)
        mask.save(mask_p)

        img_paths.append(img_p)
        mask_paths.append(mask_p)

    return img_paths, mask_paths
