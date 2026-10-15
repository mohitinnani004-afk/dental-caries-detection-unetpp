"""
Explicit, Detailed Mask Creation Pipeline for Dental Caries Segmentation.
Demonstrates step-by-step how raw dental radiographs and clinical annotations
are converted into precise ground-truth binary masks without black-box functions.

Provides:
  1. Polygon Annotation Rasterization (Clinician / LabelMe / VIA style)
  2. Parametric Coordinate & Elliptical Lesion Rasterization
  3. High-Magnification Pixel-Grid Visualization & Detailed Inset Analysis
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure output directory exists
OUTPUT_DIR = Path("outputs") / "mask_creation_detailed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_mask_from_polygon(
    image_shape: Tuple[int, int],
    polygon_vertices: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Step-by-Step Rasterization of Polygon Vertices into a Binary Mask.
    
    Parameters:
      image_shape: (height, width) of the target radiograph.
      polygon_vertices: List of (x, y) coordinates defining the caries lesion boundary.
      
    Returns:
      mask: 2D numpy array [H, W] of type uint8 (0 = background, 1 = caries lesion).
    """
    height, width = image_shape
    # 1. Initialize an empty binary mask with zeros (all background)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Rasterize the polygon boundary into the binary matrix
    # Using PIL ImageDraw polygon fill or ray-casting algorithm
    pil_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(pil_mask)
    draw.polygon(polygon_vertices, fill=1)

    mask = np.array(pil_mask, dtype=np.uint8)
    return mask


def create_mask_from_parametric_lesion(
    image_shape: Tuple[int, int],
    center: Tuple[int, int],
    radius_x: int,
    radius_y: int,
    angle_deg: float = 0.0
) -> np.ndarray:
    """
    Explicit mathematical rasterization of an elliptical caries demineralization zone.
    A pixel (x, y) belongs to the caries lesion if:
        ((x' - cx)^2 / rx^2) + ((y' - cy)^2 / ry^2) <= 1.0
    where (x', y') is rotated by angle_deg.
    """
    height, width = image_shape
    cx, cy = center
    rad = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # Create explicit coordinate grid
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Translate to origin
    dx = x_coords - cx
    dy = y_coords - cy

    # Rotate coordinates
    dx_rot = dx * cos_a + dy * sin_a
    dy_rot = -dx * sin_a + dy * cos_a

    # Elliptical distance metric
    distance = (dx_rot ** 2) / (radius_x ** 2) + (dy_rot ** 2) / (radius_y ** 2)

    # Binary mask: 1 where distance <= 1.0, 0 otherwise
    binary_mask = (distance <= 1.0).astype(np.uint8)
    return binary_mask


def combine_multiple_lesions(masks: List[np.ndarray]) -> np.ndarray:
    """
    Merges multiple individual caries lesion masks using logical OR.
    """
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        combined = np.bitwise_or(combined, m)
    return combined


def visualize_detailed_mask_pipeline(
    radiograph_path: Path,
    lesion_annotations: List[Dict],
    save_path: Path
):
    """
    Generates a detailed, publication-quality 5-panel figure showing:
      Panel 1: Original Raw Dental Radiograph
      Panel 2: Clinician Annotation Vertices & Contours
      Panel 3: Pure Rasterized Binary Mask [H, W] (0s and 255s)
      Panel 4: Fluorescent Coral Alpha-Blended Overlay
      Panel 5: High-Magnification Zoom of the Caries Boundary with Pixel Grid!
    """
    # 1. Load image
    orig_img = Image.open(radiograph_path).convert("L")
    img_np = np.array(orig_img)
    h, w = img_np.shape

    # 2. Generate explicit masks for each lesion
    individual_masks = []
    for ann in lesion_annotations:
        if ann["type"] == "polygon":
            m = create_mask_from_polygon((h, w), ann["vertices"])
        elif ann["type"] == "ellipse":
            m = create_mask_from_parametric_lesion(
                (h, w),
                center=ann["center"],
                radius_x=ann["radius_x"],
                radius_y=ann["radius_y"],
                angle_deg=ann.get("angle", 0.0)
            )
        individual_masks.append(m)

    combined_mask = combine_multiple_lesions(individual_masks)

    # 3. Setup 5-panel matplotlib figure
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), facecolor="#0e1117")
    plt.subplots_adjust(wspace=0.15)

    # Styles
    title_color = "#f0f2f6"
    accent_coral = "#ff4757"
    accent_cyan = "#00f2fe"
    accent_amber = "#ffa502"

    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#8892b0")
        for spine in ax.spines.values():
            spine.set_color("#233554")

    # Panel 1: Raw Radiograph
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("1. Raw Dental Radiograph\n(Bitewing X-Ray)", color=title_color, fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"Width: {w} px", color="#8892b0")
    axes[0].set_ylabel(f"Height: {h} px", color="#8892b0")

    # Panel 2: Annotations & Contours
    axes[1].imshow(img_np, cmap="gray")
    for i, ann in enumerate(lesion_annotations):
        if ann["type"] == "polygon":
            poly = patches.Polygon(ann["vertices"], closed=True, fill=False, edgecolor=accent_coral, linewidth=2.5)
            axes[1].add_patch(poly)
            # Plot vertices
            vx, vy = zip(*ann["vertices"])
            axes[1].scatter(vx, vy, color=accent_cyan, s=25, zorder=5)
            cx, cy = np.mean(vx), np.mean(vy)
        else:
            cx, cy = ann["center"]
            ell = patches.Ellipse((cx, cy), ann["radius_x"]*2, ann["radius_y"]*2, angle=ann.get("angle", 0),
                                  fill=False, edgecolor=accent_coral, linewidth=2.5)
            axes[1].add_patch(ell)
            axes[1].plot(cx, cy, "x", color=accent_cyan, markersize=8)

        axes[1].text(cx - 20, cy - ann.get("radius_y", 15) - 6, f"Lesion #{i+1}",
                     color=accent_amber, fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1c23", edgecolor=accent_amber, alpha=0.9))

    axes[1].set_title("2. Lesion Annotation Coordinates\n(Polygon Vertices / Contours)", color=title_color, fontsize=12, fontweight="bold")

    # Panel 3: Binary Mask Matrix
    axes[2].imshow(combined_mask * 255, cmap="gray")
    axes[2].set_title(f"3. Binary Ground-Truth Mask\n(Pixels: {np.sum(combined_mask):,} caries / {h*w:,} total)", color=title_color, fontsize=12, fontweight="bold")

    # Panel 4: Alpha-blended Overlay
    overlay_rgb = np.stack([img_np]*3, axis=-1).astype(np.float32) / 255.0
    # Blend neon coral on caries pixels: [1.0, 0.28, 0.34]
    mask_bool = combined_mask > 0
    overlay_rgb[mask_bool] = overlay_rgb[mask_bool] * 0.35 + np.array([1.0, 0.28, 0.34]) * 0.65
    axes[3].imshow(overlay_rgb)
    axes[3].set_title("4. Alpha-Blended Clinical Overlay\n(Neon Coral Caries Mask)", color=title_color, fontsize=12, fontweight="bold")

    # Panel 5: High-Magnification Pixel Grid Inset
    # Focus on first lesion center
    focus_cx = lesion_annotations[0]["center"][0] if "center" in lesion_annotations[0] else int(np.mean([p[0] for p in lesion_annotations[0]["vertices"]]))
    focus_cy = lesion_annotations[0]["center"][1] if "center" in lesion_annotations[0] else int(np.mean([p[1] for p in lesion_annotations[0]["vertices"]]))
    
    pad = 22
    y1, y2 = max(0, focus_cy - pad), min(h, focus_cy + pad)
    x1, x2 = max(0, focus_cx - pad), min(w, focus_cx + pad)

    crop_img = img_np[y1:y2, x1:x2]
    crop_mask = combined_mask[y1:y2, x1:x2]

    crop_overlay = np.stack([crop_img]*3, axis=-1).astype(np.float32) / 255.0
    crop_mask_bool = crop_mask > 0
    crop_overlay[crop_mask_bool] = crop_overlay[crop_mask_bool] * 0.35 + np.array([1.0, 0.28, 0.34]) * 0.65

    axes[4].imshow(crop_overlay, interpolation="nearest")
    # Draw pixel grid lines
    axes[4].set_xticks(np.arange(-0.5, x2 - x1, 1), minor=True)
    axes[4].set_yticks(np.arange(-0.5, y2 - y1, 1), minor=True)
    axes[4].grid(which="minor", color="cyan", linestyle="-", linewidth=0.5, alpha=0.35)
    axes[4].set_title(f"5. Zoomed Pixel Grid Inset\n(Caries Margins: {x2-x1}x{y2-y1} px)", color=title_color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Detailed mask pipeline visualization saved to: {save_path}")


def main():
    print("=== Demonstrating Detailed Dental Caries Mask Creation ===")
    sample_img_path = Path("samples") / "sample_proximal.png"

    # Define explicit lesion annotations (interproximal molar caries)
    # 1. Proximal lesion (demineralized notch at enamel contact point)
    proximal_polygon = [
        (108, 92), (116, 84), (124, 88), (128, 98),
        (126, 110), (118, 116), (110, 112), (105, 102)
    ]

    annotations = [
        {
            "type": "polygon",
            "name": "Interproximal Enamel-Dentin Caries (Tooth #19)",
            "vertices": proximal_polygon,
            "center": (116, 100),
            "severity": "Dentin D1"
        }
    ]

    # Generate and inspect mask
    mask = create_mask_from_polygon((256, 256), proximal_polygon)
    print(f"Mask Matrix Properties:")
    print(f"  Shape: {mask.shape}")
    print(f"  Data Type: {mask.dtype}")
    print(f"  Unique Values: {np.unique(mask)} (0=Sound Tooth, 1=Caries Lesion)")
    print(f"  Total Caries Pixels: {np.sum(mask):,} px")
    print(f"  Estimated Area: {np.sum(mask) * (0.08**2):.2f} mm² (at 0.08 mm/px pitch)")

    # Create visualization
    output_fig = OUTPUT_DIR / "detailed_mask_creation_walkthrough.png"
    visualize_detailed_mask_pipeline(
        radiograph_path=sample_img_path,
        lesion_annotations=annotations,
        save_path=output_fig
    )
    print("Done! You can view the generated detailed walkthrough in the outputs folder.")


if __name__ == "__main__":
    main()
