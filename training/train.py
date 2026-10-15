"""
CLI Entrypoint for training Dental Caries UNet++ segmentation model.
Usage:
    python training/train.py --epochs 40 --batch_size 8 --generate_sample
"""

import argparse
from pathlib import Path
import sys
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_MODEL_CONFIG, DEFAULT_DATA_CONFIG, DEFAULT_TRAIN_CONFIG, CHECKPOINTS_DIR, DATA_DIR
from models.unet_plus_plus import build_unet_plus_plus
from models.losses import DeepSupervisionLoss, BCEDiceLoss
from data.dataset import create_dental_dataloaders
from data.sample_generator import generate_sample_dataset
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet++ for Dental Caries Detection")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR / "dataset"), help="Path to dataset with images/ and masks/")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAIN_CONFIG.batch_size, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG.learning_rate, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=DEFAULT_DATA_CONFIG.image_size[0], help="Image spatial resolution")
    parser.add_argument("--no_deep_supervision", action="store_true", help="Disable deep supervision")
    parser.add_argument("--generate_sample", action="store_true", help="Generate synthetic dataset before training")
    parser.add_argument("--num_samples", type=int, default=48, help="Number of synthetic samples to generate if requested")
    parser.add_argument("--checkpoints_dir", type=str, default=str(CHECKPOINTS_DIR), help="Directory to save weights")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Automatically generate sample dataset if flag is passed or directory doesn't exist
    if args.generate_sample or not (data_dir / "images").exists():
        print(f"Generating {args.num_samples} synthetic dental radiographs in {data_dir}...")
        generate_sample_dataset(
            output_dir=data_dir,
            num_samples=args.num_samples,
            width=args.img_size,
            height=args.img_size
        )
        print("Synthetic dataset successfully prepared!")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = create_dental_dataloaders(
        data_dir=data_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )

    print(f"Dataset loaded: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test")

    # Instantiate UNet++
    deep_sup = not args.no_deep_supervision
    model = build_unet_plus_plus(
        in_channels=1,
        num_classes=1,
        deep_supervision=deep_sup,
        use_attention=True,
        nb_filter=[32, 64, 128, 256, 512]
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UNet++ (NestedUNet) | Trainable parameters: {total_params:,}")

    # Loss, Optimizer, Scheduler
    base_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    criterion = DeepSupervisionLoss(base_criterion=base_loss, weights=[0.1, 0.2, 0.3, 0.4]) if deep_sup else base_loss

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoints_dir,
        deep_supervision=deep_sup,
        early_stopping_patience=12
    )

    # Execute training
    history = trainer.fit(num_epochs=args.epochs)
    print(f"Training completed! Best checkpoint: {Path(args.checkpoints_dir) / 'best_model.pth'}")


if __name__ == "__main__":
    main()
