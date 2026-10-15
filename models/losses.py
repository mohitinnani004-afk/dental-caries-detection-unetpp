"""
Loss functions optimized for Dental Caries Segmentation and severe class imbalance.

Includes:
  - DiceLoss
  - Binary Focal Loss
  - Combined BCE + Dice Loss (Recommended standard)
  - Focal Tversky Loss (Customizable FN/FP penalty)
  - DeepSupervisionLoss wrapper for UNet++ nested outputs
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss with Laplace smoothing.
    Addresses severe foreground-background imbalance in dental caries X-rays.
    """
    def __init__(self, smooth: float = 1.0, p: int = 2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs.pow(self.p) + targets.pow(self.p))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for focusing gradients on hard-to-classify caries margins.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets.float() * probs + (1.0 - targets.float()) * (1.0 - probs)
        alpha_t = targets.float() * self.alpha + (1.0 - targets.float()) * (1.0 - self.alpha)

        focal_loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BCEDiceLoss(nn.Module):
    """
    Hybrid BCE + Dice Loss.
    Combines smooth gradient flow of BCE with overlap-oriented optimization of Dice.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Generalization of Dice loss allowing higher penalty on False Negatives (FN).
    Crucial for dental screening where missing a caries lesion (FN) is more detrimental than a false alert (FP).
    alpha: FP weight, beta: FN weight (beta > alpha favors recall/sensitivity).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        tp = torch.sum(probs * targets)
        fp = torch.sum(probs * (1.0 - targets))
        fn = torch.sum((1.0 - probs) * targets)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for UNet++.
    Computes weighted sum of losses across outputs [X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}].
    """
    def __init__(
        self,
        base_criterion: Optional[nn.Module] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.base_criterion = base_criterion or BCEDiceLoss()
        # Default weights give higher importance to deeper heads
        self.weights = weights or [0.1, 0.2, 0.3, 0.4]

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(preds, (list, tuple)):
            assert len(preds) == len(self.weights), (
                f"Expected {len(self.weights)} deep supervision outputs, got {len(preds)}"
            )
            total_loss = 0.0
            for pred, weight in zip(preds, self.weights):
                total_loss += weight * self.base_criterion(pred, targets)
            return total_loss
        else:
            return self.base_criterion(preds, targets)
