"""
Evaluation pipeline for Dental Caries UNet++ segmentation.
Calculates pixel-level and lesion-level clinical metrics and generates visual comparison reports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.metrics import MetricTracker, compute_lesion_level_metrics


class Evaluator:
    """
    Evaluates UNet++ on dental radiograph test partitions.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        min_lesion_area: int = 15
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.min_lesion_area = min_lesion_area

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        save_visualizations: bool = True,
        output_dir: Union[str, Path] = "outputs/evaluation"
    ) -> Dict[str, Union[float, Dict]]:
        out_path = Path(output_dir)
        if save_visualizations:
            vis_dir = out_path / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

        tracker = MetricTracker(threshold=self.threshold)
        all_pred_masks = []
        all_target_masks = []
        saved_samples_count = 0

        for images, masks, filenames in test_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            outputs = self.model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            probs = torch.sigmoid(outputs)
            tracker.update(outputs, masks)

            # Store binary masks as numpy arrays for lesion-level metrics
            prob_np = probs.cpu().numpy().squeeze(axis=1) # [B, H, W]
            mask_np = masks.cpu().numpy().squeeze(axis=1) # [B, H, W]

            for b in range(images.size(0)):
                p_bin = (prob_np[b] >= self.threshold).astype(np.uint8)
                t_bin = (mask_np[b] >= 0.5).astype(np.uint8)
                all_pred_masks.append(p_bin)
                all_target_masks.append(t_bin)

                # Save up to 10 visual comparison plates
                if save_visualizations and saved_samples_count < 10:
                    orig_img = images[b, 0].cpu().numpy()
                    self._save_comparison_plate(
                        orig_img=orig_img,
                        prob_map=prob_np[b],
                        pred_bin=p_bin,
                        gt_bin=t_bin,
                        filename=filenames[b],
                        save_path=vis_dir / f"eval_{filenames[b]}"
                    )
                    saved_samples_count += 1

        pixel_metrics = tracker.get_metrics()

        # Compute lesion-level metrics
        if all_pred_masks and all_target_masks:
            lesion_metrics = compute_lesion_level_metrics(
                np.array(all_pred_masks),
                np.array(all_target_masks),
                min_iou=0.25,
                min_area=self.min_lesion_area
            )
        else:
            lesion_metrics = {}

        summary = {
            "threshold": self.threshold,
            "pixel_metrics": pixel_metrics,
            "lesion_metrics": lesion_metrics,
            "test_sample_count": len(all_pred_masks)
        }

        # Save summary report
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "evaluation_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _save_comparison_plate(
        self,
        orig_img: np.ndarray,
        prob_map: np.ndarray,
        pred_bin: np.ndarray,
        gt_bin: np.ndarray,
        filename: str,
        save_path: Path
    ):
        """
        Creates a side-by-side diagnostic comparison panel:
        [1. Original Radiograph | 2. Ground Truth Caries | 3. Predicted Probability Heatmap | 4. Predicted Binary Overlay]
        """
        h, w = orig_img.shape

        # Normalize original image from [-1, 1] to [0, 255]
        norm_img = ((orig_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        img_rgb = Image.fromarray(norm_img).convert("RGB")

        # Ground truth overlay (Green)
        gt_overlay = img_rgb.copy()
        gt_pixels = gt_overlay.load()
        for y in range(h):
            for x in range(w):
                if gt_bin[y, x] > 0:
                    r, g, b = gt_pixels[x, y]
                    gt_pixels[x, y] = (int(r * 0.4), int(g * 0.4 + 150), int(b * 0.4))

        # Probability Heatmap (Color mapped: Black -> Blue -> Yellow -> Red)
        heat_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                p = float(prob_map[y, x])
                if p < 0.25:
                    heat_arr[y, x] = [0, 0, int(p * 4 * 255)]
                elif p < 0.65:
                    t = (p - 0.25) / 0.4
                    heat_arr[y, x] = [int(t * 255), int(t * 255), int((1 - t) * 255)]
                else:
                    t = (p - 0.65) / 0.35
                    heat_arr[y, x] = [255, int((1 - t) * 200), 0]
        heat_img = Image.fromarray(heat_arr)

        # Predicted Overlay (Red/Orange)
        pred_overlay = img_rgb.copy()
        pred_pixels = pred_overlay.load()
        for y in range(h):
            for x in range(w):
                if pred_bin[y, x] > 0:
                    r, g, b = pred_pixels[x, y]
                    pred_pixels[x, y] = (int(r * 0.3 + 175), int(g * 0.3), int(b * 0.3))

        # Create combined 4-panel image
        panel_w = w * 4 + 30
        panel_h = h + 60
        plate = Image.new("RGB", (panel_w, panel_h), (25, 27, 34))

        draw = ImageDraw.Draw(plate)
        draw.text((15, 12), f"Dental X-Ray: {filename}", fill=(240, 240, 240))

        # Paste sub-panels
        draw.text((10, 36), "1. Original Radiograph", fill=(180, 180, 190))
        plate.paste(img_rgb, (10, 52))

        draw.text((w + 18, 36), "2. Ground Truth Mask", fill=(100, 220, 120))
        plate.paste(gt_overlay, (w + 18, 52))

        draw.text((w * 2 + 24, 36), "3. Probability Heatmap", fill=(255, 180, 70))
        plate.paste(heat_img, (w * 2 + 24, 52))

        draw.text((w * 3 + 30, 36), "4. UNet++ Prediction", fill=(255, 90, 90))
        plate.paste(pred_overlay, (w * 3 + 30, 52))

        plate.save(save_path)
