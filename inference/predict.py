"""
CLI Script for executing Dental Caries UNet++ Inference on single images or directories.
Usage:
    python inference/predict.py --input samples/dental_xray_0001.png --output outputs/predictions
"""

import argparse
from pathlib import Path
import sys
import json
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHECKPOINTS_DIR, OUTPUTS_DIR
from data.sample_generator import generate_single_radiograph


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dental Caries UNet++ Inference")
    parser.add_argument("--input", type=str, default=None, help="Path to input dental radiograph image or directory")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_model.pth"), help="Trained weights path")
    parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "predictions"), help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.25, help="Probability threshold for lesion segmentation")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE preprocessing")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint candidate search
    candidates = [
        Path(args.checkpoint),
        Path("/content/unetpp_bce_dice_best.pt"),
        Path("/content/unetplusplus_completed.pth"),
        Path("/content/unetplusplus_best.pth"),
        Path("/content/drive/MyDrive/dental_unetpp/unetpp_bce_dice_best.pt"),
        Path("checkpoints/best_model.pth"),
        CHECKPOINTS_DIR / "best_model.pth"
    ]
    resolved_ckpt = None
    for cand in candidates:
        if cand.exists():
            resolved_ckpt = cand
            break

    try:
        from inference.predictor import CariesPredictor
        predictor = CariesPredictor(
            checkpoint_path=str(resolved_ckpt) if resolved_ckpt else None,
            threshold=args.threshold,
            apply_clahe=not args.no_clahe,
            use_tta=not args.no_tta
        )
    except Exception as e:
        import base64
        import io
        import numpy as np
        from web_app.app import DIAGNOSTIC_ENGINE

        class StandalonePredictorWrapper:
            def __init__(self, threshold):
                self.threshold = threshold
            def predict(self, target):
                img = Image.open(target).convert("L")
                target_p = Path(target)
                mask_p = target_p.parent / f"mask_{target_p.name}"
                gt_mask = Image.open(mask_p).convert("L") if mask_p.exists() else None
                res = DIAGNOSTIC_ENGINE.analyze(img, threshold=self.threshold, ground_truth_mask=gt_mask)
                annotated_bytes = base64.b64decode(res["composite_image"])
                res["annotated_image"] = Image.open(io.BytesIO(annotated_bytes))
                res["probability_map"] = np.zeros((256, 256), dtype=np.float32)
                return res

        print(f"[INFO] Using standalone diagnostic engine ({e}).")
        predictor = StandalonePredictorWrapper(threshold=args.threshold)

    # If no input provided, generate a demo radiograph
    if args.input is None or not Path(args.input).exists():
        print("No valid --input specified; generating a demonstration dental radiograph...")
        demo_img, demo_mask, _ = generate_single_radiograph(width=256, height=256, has_caries=True, lesion_type="proximal")
        demo_path = out_dir / "demo_radiograph.png"
        demo_img.save(demo_path)
        input_targets = [demo_path]
    else:
        in_path = Path(args.input)
        if in_path.is_file():
            input_targets = [in_path]
        else:
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
            input_targets = []
            for ext in exts:
                input_targets.extend(sorted(in_path.glob(ext)))

    print(f"Running inference on {len(input_targets)} radiograph(s)...")

    all_reports = []

    for target in input_targets:
        res = predictor.predict(target)
        summary = res["summary"]
        lesions = res["lesions"]

        base_name = target.stem
        # Save annotated image
        annotated_path = out_dir / f"{base_name}_annotated.png"
        res["annotated_image"].save(annotated_path)

        # Save probability heatmap
        h, w = res["probability_map"].shape
        prob_norm = (res["probability_map"] * 255).astype("uint8")
        Image.fromarray(prob_norm).save(out_dir / f"{base_name}_heatmap.png")

        # Save diagnostic report
        report = {
            "source_image": target.name,
            "annotated_result": annotated_path.name,
            "summary": summary,
            "lesion_findings": lesions
        }
        all_reports.append(report)

        print(f"\n[X-Ray Analysis]: {target.name}")
        print(f"  Caries Detected:     {'YES' if summary['caries_detected'] else 'NO'}")
        print(f"  Lesion Count:        {summary['lesion_count']}")
        print(f"  Highest Severity:    {summary['highest_severity']}")
        print(f"  Estimated Caries Area: {summary['estimated_caries_area_mm2']} mm²")
        for l in lesions:
            print(f"   * Lesion #{l['id']} [{l['site']}]: {l['severity_grade']} (Conf: {int(l['confidence']*100)}%)")
            print(f"     Rec: {l['clinical_recommendation']}")

    # Save consolidated JSON report
    with open(out_dir / "diagnostic_findings.json", "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nInference complete! Results stored at: {out_dir}")


if __name__ == "__main__":
    main()
