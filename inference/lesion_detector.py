"""
Clinical Lesion Detection and Anatomical Severity Classification.
Analyzes UNet++ segmentation masks to isolate individual caries lesions,
compute bounding boxes, classify anatomical sites, and estimate radiographic depth/severity.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CariesLesion:
    id: int
    site: str               # "Occlusal", "Proximal", "Cervical"
    severity_grade: str     # "E1 (Outer Enamel)", "E2 (Inner Enamel)", "D1 (Outer Dentin)", "D2 (Middle Dentin)", "D3 (Deep/Pulpal Risk)"
    confidence: float
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    centroid: Tuple[int, int]       # (cx, cy)
    area_pixels: int
    estimated_area_mm2: float
    clinical_recommendation: str


class LesionDetector:
    """
    Performs morphological and connected-component analysis on binary segmentation masks
    to extract clinically meaningful caries findings.
    """
    def __init__(
        self,
        min_area: int = 15,
        connectivity: int = 8,
        pixel_pitch_mm: float = 0.08  # Approx 0.08 mm per pixel on 256x256 dental radiograph
    ):
        self.min_area = min_area
        self.connectivity = connectivity
        self.pixel_pitch_mm = pixel_pitch_mm

    def detect_lesions(
        self,
        binary_mask: np.ndarray,
        probability_map: np.ndarray,
        image_shape: Tuple[int, int] = (256, 256)
    ) -> List[CariesLesion]:
        """
        Extracts individual lesions from binary mask and probability map.
        """
        try:
            from scipy.ndimage import label, find_objects
            labeled, num_features = label(binary_mask > 0)
            slices = find_objects(labeled)
        except ImportError:
            # Fallback pure numpy flood-fill / bounding box algorithm
            labeled, slices, num_features = self._simple_connected_components(binary_mask)

        lesions: List[CariesLesion] = []
        img_h, img_w = image_shape

        for i, s in enumerate(slices):
            if s is None:
                continue

            comp_mask = (labeled == (i + 1))
            area = int(np.sum(comp_mask))

            # Filter out tiny noise specks
            if area < self.min_area:
                continue

            # Bounding box
            y_indices, x_indices = np.where(comp_mask)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

            # Centroid
            cx = int(np.mean(x_indices))
            cy = int(np.mean(y_indices))

            # Average confidence inside lesion
            mean_prob = float(np.mean(probability_map[comp_mask]))

            # Estimated real-world area in mm²
            area_mm2 = round(area * (self.pixel_pitch_mm ** 2), 2)

            # Anatomical site and severity estimation
            site, severity, rec = self._classify_lesion_depth_and_site(
                cx=cx,
                cy=cy,
                bbox=bbox,
                area=area,
                img_w=img_w,
                img_h=img_h
            )

            lesion = CariesLesion(
                id=len(lesions) + 1,
                site=site,
                severity_grade=severity,
                confidence=round(mean_prob, 3),
                bbox=bbox,
                centroid=(cx, cy),
                area_pixels=area,
                estimated_area_mm2=area_mm2,
                clinical_recommendation=rec
            )
            lesions.append(lesion)

        return lesions

    def _classify_lesion_depth_and_site(
        self,
        cx: int,
        cy: int,
        bbox: Tuple[int, int, int, int],
        area: int,
        img_w: int,
        img_h: int
    ) -> Tuple[str, str, str]:
        """
        Classifies anatomical site and ICDAS-equivalent radiographic depth.
        """
        _, _, w, h = bbox
        rel_y = cy / img_h
        rel_x = cx / img_w

        # Anatomical site based on vertical and horizontal dental geometry
        if rel_y < 0.32:
            site = "Occlusal"
        elif rel_y > 0.65:
            site = "Cervical / Root"
        else:
            site = "Proximal (Interproximal)"

        # Severity estimation based on lesion extent and penetration depth
        max_dim = max(w, h)

        if max_dim <= 8:
            severity = "E1 (Outer Enamel Caries)"
            rec = "Non-operative remineralization therapy (high-fluoride varnish/gel)."
        elif max_dim <= 14:
            severity = "E2 (Inner Enamel Caries)"
            rec = "Remineralization monitoring or resin infiltration (Icon)."
        elif max_dim <= 22:
            severity = "D1 (Outer 1/3 Dentin Caries)"
            rec = "Conservative restorative intervention (composite restoration)."
        elif max_dim <= 32:
            severity = "D2 (Middle 1/3 Dentin Caries)"
            rec = "Restorative therapy required; caries excavation and restoration."
        else:
            severity = "D3 (Deep Dentin / Pulpal Risk)"
            rec = "Urgent endodontic/restorative evaluation; indirect pulp capping or pulpotomy risk."

        return site, severity, rec

    def _simple_connected_components(self, mask: np.ndarray):
        """Simple label implementation if scipy is not installed."""
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        labeled = np.zeros_like(mask, dtype=int)
        slices = []
        current_label = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] > 0 and not visited[r, c]:
                    current_label += 1
                    queue = [(r, c)]
                    visited[r, c] = True
                    labeled[r, c] = current_label
                    min_r, max_r = r, r
                    min_c, max_c = c, c

                    while queue:
                        cr, cc = queue.pop(0)
                        min_r = min(min_r, cr)
                        max_r = max(max_r, cr)
                        min_c = min(min_c, cc)
                        max_c = max(max_c, cc)

                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if mask[nr, nc] > 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    labeled[nr, nc] = current_label
                                    queue.append((nr, nc))

                    slices.append((slice(min_r, max_r + 1), slice(min_c, max_c + 1)))

        return labeled, slices, current_label
