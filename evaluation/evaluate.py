"""
CLI Entrypoint for running full test set evaluation on a trained UNet++ checkpoint.
Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/dataset
"""

import argparse
from pathlib import Path
import sys
try:
    import torch
    from models.unet_plus_plus import build_unet_plus_plus
    from evaluation.evaluator import Evaluator
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_DATA_CONFIG, OUTPUTS_DIR, CHECKPOINTS_DIR, DATA_DIR
from data.dataset import create_dental_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet++ on Dental Caries Test Set")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_model.pth"), help="Path to checkpoint .pth")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR / "dataset"), help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR / "evaluation"), help="Where to save results")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold for binarization")
    parser.add_argument("--img_size", type=int, default=DEFAULT_DATA_CONFIG.image_size[0], help="Image size")
    return parser.parse_args()


def main():
    args = parse_args()
    if not TORCH_AVAILABLE:
        print("\n" + "=" * 65)
        print("  UNet++ EVALUATION NOTICE")
        print("=" * 65)
        print("[INFO] PyTorch is required to run neural network evaluation.")
        print("To evaluate your trained checkpoint:")
        print("  1. In Google Colab: Run Cell 8 in notebooks/dental_caries_colab_training.ipynb")
        print("  2. Locally: Run 'pip install torch torchvision' in your virtual environment.")
        print("=" * 65 + "\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Dental Caries UNet++ on {device} ===")

    # Initialize Model
    model = build_unet_plus_plus(
        in_channels=1,
        num_classes=1,
        deep_supervision=True,
        nb_filter=[32, 64, 128, 256, 512]
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        epoch_info = state.get("epoch", "N/A") if isinstance(state, dict) else "N/A"
        print(f"Loaded checkpoint from: {ckpt_path} (Trained Epoch {epoch_info})")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Evaluating with initial weights for pipeline validation.")

    # Dataloaders
    _, _, test_loader = create_dental_dataloaders(
        data_dir=args.data_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=4
    )

    evaluator = Evaluator(model=model, device=device, threshold=args.threshold)
    results = evaluator.evaluate(
        test_loader=test_loader,
        save_visualizations=True,
        output_dir=args.output_dir
    )

    pix = results["pixel_metrics"]
    les = results.get("lesion_metrics", {})

    print("\n" + "=" * 50)
    print("           EVALUATION SUMMARY RESULTS")
    print("=" * 50)
    print(f"  Dice Similarity (DSC): {pix.get('dice', 0.0):.4f}")
    print(f"  Intersection over Union: {pix.get('iou', 0.0):.4f}")
    print(f"  Pixel Sensitivity:     {pix.get('sensitivity', 0.0):.4f}")
    print(f"  Pixel Specificity:     {pix.get('specificity', 0.0):.4f}")
    print(f"  Pixel Precision:       {pix.get('precision', 0.0):.4f}")
    print(f"  Pixel Accuracy:        {pix.get('accuracy', 0.0):.4f}")
    if les:
        print("-" * 50)
        print(f"  Lesion Detection F1:   {les.get('lesion_f1', 0.0):.4f}")
        print(f"  Lesion Recall:         {les.get('lesion_recall', 0.0):.4f}")
        print(f"  Lesion Precision:      {les.get('lesion_precision', 0.0):.4f}")
        print(f"  True Detected Lesions: {les.get('detected_lesions', 0)} / {les.get('total_true_lesions', 0)}")
    print("=" * 50)
    print(f"Detailed report & visualizations saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
