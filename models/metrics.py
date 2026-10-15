"""
Segmentation and Detection Metrics for Dental Caries Evaluation.
Includes pixel-level metrics (Dice, IoU, Sensitivity, Specificity, Precision)
and lesion-level connected component detection metrics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Computes (TP, FP, FN, TN) for binary segmentation.
    """
    if preds.ndim > 2:
        probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1.0) else preds
        pred_bin = (probs >= threshold).long().view(-1)
    else:
        pred_bin = (preds >= threshold).long().view(-1)

    target_bin = (targets >= 0.5).long().view(-1)

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()

    return tp, fp, fn, tn


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Dice Similarity Coefficient (DSC / F1 score)."""
    tp, fp, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Intersection over Union (IoU / Jaccard Index)."""
    tp, fp, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (tp + eps) / (tp + fp + fn + eps)


def sensitivity_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Sensitivity / Recall (True Positive Rate)."""
    tp, _, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Specificity (True Negative Rate)."""
    _, fp, _, tn = compute_confusion_matrix(preds, targets, threshold)
    return (tn + eps) / (tn + fp + eps)


def precision_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Precision / Positive Predictive Value."""
    tp, fp, _, _ = compute_confusion_matrix(preds, targets, threshold)
    return (tp + eps) / (tp + fp + eps)


class MetricTracker:
    """
    Accumulates batch-level TP, FP, FN, TN to compute dataset-wide metrics accurately
    rather than unweighted averaging of batch metrics.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.sample_count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        tp, fp, fn, tn = compute_confusion_matrix(preds, targets, self.threshold)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.sample_count += targets.size(0)

    def get_metrics(self, eps: float = 1e-7) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        sensitivity = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        precision = (tp + eps) / (tp + fp + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        return {
            "dice": float(dice),
            "iou": float(iou),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "accuracy": float(accuracy),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }


def compute_lesion_level_metrics(
    pred_masks: np.ndarray,
    target_masks: np.ndarray,
    min_iou: float = 0.3,
    min_area: int = 15
) -> Dict[str, float]:
    """
    Evaluates caries detection at individual lesion level using connected components.
    A predicted lesion is a TP if it overlaps a ground-truth lesion with IoU >= min_iou.
    """
    from scipy.ndimage import label

    total_gt_lesions = 0
    total_pred_lesions = 0
    true_positives = 0

    for i in range(len(pred_masks)):
        p_mask = pred_masks[i] > 0.5
        t_mask = target_masks[i] > 0.5

        # Label individual connected components (caries lesions)
        p_labeled, p_num = label(p_mask)
        t_labeled, t_num = label(t_mask)

        # Filter out micro-noise
        valid_p = []
        for p_idx in range(1, p_num + 1):
            if np.sum(p_labeled == p_idx) >= min_area:
                valid_p.append(p_idx)

        valid_t = []
        for t_idx in range(1, t_num + 1):
            if np.sum(t_labeled == t_idx) >= min_area:
                valid_t.append(t_idx)

        total_gt_lesions += len(valid_t)
        total_pred_lesions += len(valid_p)

        # Match predicted lesions with ground-truth lesions
        matched_gt = set()
        for p_idx in valid_p:
            p_comp = (p_labeled == p_idx)
            best_iou = 0.0
            best_t = None
            for t_idx in valid_t:
                if t_idx in matched_gt:
                    continue
                t_comp = (t_labeled == t_idx)
                inter = np.sum(p_comp & t_comp)
                union = np.sum(p_comp | t_comp)
                iou = inter / (union + 1e-7)
                if iou > best_iou:
                    best_iou = iou
                    best_t = t_idx

            if best_iou >= min_iou and best_t is not None:
                true_positives += 1
                matched_gt.add(best_t)

    false_positives = total_pred_lesions - true_positives
    false_negatives = total_gt_lesions - true_positives

    precision = true_positives / (total_pred_lesions + 1e-7)
    recall = true_positives / (total_gt_lesions + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "detected_lesions": true_positives,
        "false_positive_lesions": false_positives,
        "missed_lesions": false_negatives,
        "total_true_lesions": total_gt_lesions
    }
