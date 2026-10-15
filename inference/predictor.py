"""
High-level Inference Engine for Dental Caries UNet++.
Performs preprocessing, model prediction, TTA, lesion extraction, and diagnostic rendering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from models.unet_plus_plus import build_unet_plus_plus, UNetPlusPlus
from data.dataset import apply_clahe_preprocessing
from .lesion_detector import LesionDetector, CariesLesion, asdict


class CariesPredictor:
    """
    Diagnostic Predictor integrating UNet++ with morphological lesion analysis.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.50,
        apply_clahe: bool = True,
        use_tta: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.threshold = threshold
        self.apply_clahe = apply_clahe
        self.use_tta = use_tta
        self.detector = LesionDetector(min_area=15)

        # Initialize Model using intelligent checkpoint loader
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                from models.unet_plus_plus import load_unet_plus_plus_checkpoint
                self.model, _ = load_unet_plus_plus_checkpoint(checkpoint_path, device=self.device)
                print(f"[OK] Loaded Caries UNet++ weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[WARN] Failed loading checkpoint with helper ({e}); trying standard init.")
                self.model = build_unet_plus_plus(
                    in_channels=1,
                    num_classes=1,
                    deep_supervision=False,
                    nb_filter=[32, 64, 128, 256, 512]
                ).to(self.device)
        else:
            self.model = build_unet_plus_plus(
                in_channels=1,
                num_classes=1,
                deep_supervision=False,
                nb_filter=[32, 64, 128, 256, 512]
            ).to(self.device)
            print("Notice: No trained checkpoint supplied; using initialized weights.")
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        custom_threshold: Optional[float] = None
    ) -> Dict:
        """
        Executes inference on an input dental radiograph.
        Returns:
          - probability_map: 2D float ndarray [0.0, 1.0]
          - binary_mask: 2D uint8 ndarray {0, 1}
          - lesions: List of CariesLesion objects
          - annotated_image: PIL Image with overlays & bounding boxes
          - summary: Clinical summary metrics
        """
        thresh = custom_threshold if custom_threshold is not None else self.threshold

        # Load & format input image
        if isinstance(image_input, (str, Path)):
            orig_pil = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input.astype(np.uint8)).convert("L")
        else:
            orig_pil = image_input.convert("L")

        orig_w, orig_h = orig_pil.size

        # Preprocessing: CLAHE
        if self.apply_clahe:
            proc_pil = apply_clahe_preprocessing(orig_pil)
        else:
            proc_pil = orig_pil.copy()

        # Resize to network input dimension
        resized_pil = proc_pil.resize(self.image_size, Image.BILINEAR)

        # Normalize [-1.0, 1.0]
        img_np = np.array(resized_pil, dtype=np.float32) / 255.0
        img_norm = (img_np - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(img_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        probs = torch.sigmoid(logits)

        # Test-Time Augmentation (Horizontal Flip)
        if self.use_tta:
            flipped_tensor = torch.flip(img_tensor, dims=[3])
            flip_logits = self.model(flipped_tensor)
            if isinstance(flip_logits, (list, tuple)):
                flip_logits = flip_logits[-1]
            flip_probs = torch.flip(torch.sigmoid(flip_logits), dims=[3])
            probs = 0.5 * (probs + flip_probs)

        # Interpolate back to original resolution if needed
        if (orig_w, orig_h) != self.image_size:
            probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        prob_map = probs.squeeze().cpu().numpy()
        binary_mask = (prob_map >= thresh).astype(np.uint8)

        # Lesion detection & classification
        lesions = self.detector.detect_lesions(
            binary_mask=binary_mask,
            probability_map=prob_map,
            image_shape=(orig_h, orig_w)
        )

        # Render diagnostic overlay
        annotated_img = self._render_diagnostic_overlay(
            orig_pil=orig_pil,
            prob_map=prob_map,
            binary_mask=binary_mask,
            lesions=lesions
        )

        total_caries_pixels = int(np.sum(binary_mask))
        summary = {
            "caries_detected": len(lesions) > 0,
            "lesion_count": len(lesions),
            "total_caries_pixels": total_caries_pixels,
            "estimated_caries_area_mm2": round(total_caries_pixels * (self.detector.pixel_pitch_mm ** 2), 2),
            "max_confidence": float(np.max(prob_map)),
            "applied_threshold": thresh,
            "highest_severity": max([l.severity_grade for l in lesions], default="None (Healthy)")
        }

        return {
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "lesions": [asdict(l) for l in lesions],
            "lesion_objects": lesions,
            "annotated_image": annotated_img,
            "summary": summary
        }

    def _render_diagnostic_overlay(
        self,
        orig_pil: Image.Image,
        prob_map: np.ndarray,
        binary_mask: np.ndarray,
        lesions: List[CariesLesion]
    ) -> Image.Image:
        """
        Creates an annotated clinical diagnosis view:
          - Grayscale background
          - Translucent neon-coral overlay on caries pixels
          - Bounding boxes around individual detected cavities
          - Badges indicating Lesion ID and Severity
        """
        orig_rgb = orig_pil.convert("RGB")
        overlay = orig_rgb.copy()
        pixels = overlay.load()
        w, h = orig_pil.size

        # Tint detected caries pixels (neon coral red: RGB 255, 60, 60)
        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] > 0:
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        min(255, int(r * 0.35 + 255 * 0.65)),
                        int(g * 0.35),
                        int(b * 0.35)
                    )

        draw = ImageDraw.Draw(overlay)

        # Draw bounding boxes and labels for each lesion
        palette = [
            (255, 75, 75),   # Coral red
            (255, 170, 0),   # Amber
            (0, 220, 255),   # Cyan
            (255, 0, 180),   # Magenta
        ]

        for i, lesion in enumerate(lesions):
            col = palette[i % len(palette)]
            x, y, bw, bh = lesion.bbox

            # Draw glowing bounding box
            draw.rectangle([x - 2, y - 2, x + bw + 2, y + bh + 2], outline=col, width=2)

            # Badge text
            badge_text = f"#{lesion.id} {lesion.site.split()[0]} ({int(lesion.confidence * 100)}%)"

            # Badge background
            tag_y = max(4, y - 18)
            draw.rectangle([x - 2, tag_y, x + min(bw + 60, 140), tag_y + 16], fill=(20, 20, 25))
            draw.rectangle([x - 2, tag_y, x + min(bw + 60, 140), tag_y + 16], outline=col, width=1)
            draw.text((x + 2, tag_y + 1), badge_text, fill=(255, 255, 255))

        return overlay
