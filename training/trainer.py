"""
Training Engine for UNet++ Dental Caries Segmentation.
Supports deep supervision, mixed precision training (AMP), learning rate scheduling,
early stopping, and comprehensive validation tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau

from models.losses import DeepSupervisionLoss, BCEDiceLoss
from models.metrics import MetricTracker


class Trainer:
    """
    Manages the complete training and validation cycle for UNet++.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        deep_supervision: bool = True,
        mixed_precision: bool = True,
        early_stopping_patience: int = 12
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or DeepSupervisionLoss(base_criterion=BCEDiceLoss())
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.deep_supervision = deep_supervision
        self.mixed_precision = mixed_precision and (self.device.type == "cuda")
        self.early_stopping_patience = early_stopping_patience

        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_sensitivity": [],
            "val_precision": [],
            "learning_rate": []
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = len(self.train_loader)

        for step, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        tracker = MetricTracker(threshold=0.5)
        batch_count = len(self.val_loader)

        for images, masks, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            outputs = self.model(images)

            # In eval mode with deep_supervision, model returns final full resolution map
            if isinstance(outputs, (list, tuple)):
                eval_pred = outputs[-1]
            else:
                eval_pred = outputs

            loss = self.criterion(eval_pred, masks)
            total_loss += loss.item()

            tracker.update(eval_pred, masks)

        metrics = tracker.get_metrics()
        metrics["val_loss"] = total_loss / max(batch_count, 1)
        return metrics

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        print(f"=== Starting Training on device: {self.device} ===")
        print(f"Epochs: {num_epochs} | Deep Supervision: {self.deep_supervision} | AMP: {self.mixed_precision}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()

            val_loss = val_metrics["val_loss"]
            val_dice = val_metrics["dice"]
            val_iou = val_metrics["iou"]
            val_sens = val_metrics["sensitivity"]
            val_prec = val_metrics["precision"]

            # Current learning rate
            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)
            self.history["val_sensitivity"].append(val_sens)
            self.history["val_precision"].append(val_prec)
            self.history["learning_rate"].append(curr_lr)

            duration = time.time() - ep_start
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] ({duration:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"LR: {curr_lr:.2e}"
            )

            # Checkpoint best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                }, best_path)
                print(f" -> Best model saved with Val Dice: {val_dice:.4f}")
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            last_path = self.checkpoint_dir / "last_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
            }, last_path)

            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience}).")
                break

        total_duration = time.time() - start_time
        print(f"=== Training Completed in {total_duration / 60:.2f} mins | Best Val Dice: {self.best_val_dice:.4f} ===")

        # Save history log
        history_file = self.checkpoint_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
