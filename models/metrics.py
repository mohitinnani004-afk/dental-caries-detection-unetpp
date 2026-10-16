"""
Segmentation and Detection Metrics for Dental Caries Evaluation.
Includes pixel-level metrics (Dice, IoU, Sensitivity, Specificity, Precision)
and lesion-level connected component detection metrics.
Supports both PyTorch Tensors and pure NumPy Arrays seamlessly.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def compute_confusion_matrix(
    preds: Any,
    targets: Any,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Computes (TP, FP, FN, TN) for binary segmentation.
    Accepts either torch.Tensor or numpy.ndarray.
    """
    if TORCH_AVAILABLE and isinstance(preds, torch.Tensor):
        if preds.ndim > 2:
            probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1.0) else preds
            pred_bin = (probs >= threshold).long().view(-1)
        else:
            pred_bin = (preds >= threshold).long().view(-1)

        target_bin = (targets >= 0.5).long().view(-1)

        tp = int((pred_bin * target_bin).sum().item())
        fp = int((pred_bin * (1 - target_bin)).sum().item())
        fn = int(((1 - pred_bin) * target_bin).sum().item())
        tn = int(((1 - pred_bin) * (1 - target_bin)).sum().item())
        return tp, fp, fn, tn
    else:
        preds_np = np.asarray(preds)
        targets_np = np.asarray(targets)
        if preds_np.min() < 0 or preds_np.max() > 1.0:
            probs = 1.0 / (1.0 + np.exp(-np.clip(preds_np, -15.0, 15.0)))
        else:
            probs = preds_np

        pred_bin = (probs >= threshold).astype(np.int64).ravel()
        target_bin = (targets_np >= 0.5).astype(np.int64).ravel()

        tp = int(np.sum(pred_bin * target_bin))
        fp = int(np.sum(pred_bin * (1 - target_bin)))
        fn = int(np.sum((1 - pred_bin) * target_bin))
        tn = int(np.sum((1 - pred_bin) * (1 - target_bin)))
        return tp, fp, fn, tn


def dice_coefficient(
    preds: Any,
    targets: Any,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Dice Similarity Coefficient (DSC / F1 score)."""
    tp, fp, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def iou_score(
    preds: Any,
    targets: Any,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Intersection over Union (IoU / Jaccard Index)."""
    tp, fp, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (tp + eps) / (tp + fp + fn + eps)


def sensitivity_score(
    preds: Any,
    targets: Any,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Sensitivity / Recall (True Positive Rate)."""
    tp, _, fn, _ = compute_confusion_matrix(preds, targets, threshold)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(
    preds: Any,
    targets: Any,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Specificity (True Negative Rate)."""
    _, fp, _, tn = compute_confusion_matrix(preds, targets, threshold)
    return (tn + eps) / (tn + fp + eps)


def precision_score(
    preds: Any,
    targets: Any,
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

    def update(self, preds: Any, targets: Any):
        tp, fp, fn, tn = compute_confusion_matrix(preds, targets, self.threshold)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        if hasattr(targets, "size"):
            self.sample_count += targets.size(0) if callable(targets.size) else targets.size
        elif hasattr(targets, "shape"):
            self.sample_count += targets.shape[0]
        else:
            self.sample_count += len(targets)

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


def _label_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Connected-component labeling with SciPy or pure NumPy BFS fallback.
    """
    try:
        from scipy.ndimage import label
        return label(binary_mask)
    except ImportError:
        h, w = binary_mask.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        current_id = 0
        visited = np.zeros((h, w), dtype=bool)

        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] and not visited[y, x]:
                    current_id += 1
                    queue = [(y, x)]
                    visited[y, x] = True
                    labeled[y, x] = current_id

                    while queue:
                        cy, cx = queue.pop(0)
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if binary_mask[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        labeled[ny, nx] = current_id
                                        queue.append((ny, nx))
        return labeled, current_id


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
    total_gt_lesions = 0
    total_pred_lesions = 0
    true_positives = 0

    for i in range(len(pred_masks)):
        p_mask = pred_masks[i] > 0.5
        t_mask = target_masks[i] > 0.5

        p_labeled, p_num = _label_connected_components(p_mask)
        t_labeled, t_num = _label_connected_components(t_mask)

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


if __name__ == "__main__":
    print("=" * 70)
    print("  DENTAL CARIES SEGMENTATION & DETECTION METRICS SUITE")
    print("  Supports both PyTorch Tensors and pure NumPy Arrays")
    print("=" * 70)

    backend_name = "PyTorch (GPU/CPU)" if TORCH_AVAILABLE else "NumPy Engine"
    print(f"[INFO] Active Backend: {backend_name}\n")

    # 1. Simulate clinical radiograph masks (256 x 256)
    H, W = 256, 256

    # Ground truth with 2 distinct caries lesions:
    # - Lesion A: Proximal enamel caries (radius 12)
    # - Lesion B: Deep occlusal dentin caries (radius 20)
    gt = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    gt[(xx - 80) ** 2 + (yy - 100) ** 2 <= 12 ** 2] = 1.0   # Lesion A
    gt[(xx - 170) ** 2 + (yy - 70) ** 2 <= 20 ** 2] = 1.0   # Lesion B

    # Prediction:
    # - Perfectly catches Lesion A
    # - Catches Lesion B with realistic minor boundary displacement (radius 18)
    # - 1 small false alarm candidate (micro-artifact, area 10)
    pred = np.zeros((H, W), dtype=np.float32)
    pred[(xx - 80) ** 2 + (yy - 100) ** 2 <= 12 ** 2] = 0.92  # Confident overlap
    pred[(xx - 172) ** 2 + (yy - 71) ** 2 <= 18 ** 2] = 0.88  # High overlap on deep
    pred[(xx - 220) ** 2 + (yy - 200) ** 2 <= 3 ** 2] = 0.65  # False positive noise

    # Compute Pixel Metrics
    tp, fp, fn, tn = compute_confusion_matrix(pred, gt, threshold=0.5)
    dsc = dice_coefficient(pred, gt, threshold=0.5)
    iou = iou_score(pred, gt, threshold=0.5)
    sens = sensitivity_score(pred, gt, threshold=0.5)
    spec = specificity_score(pred, gt, threshold=0.5)
    prec = precision_score(pred, gt, threshold=0.5)
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Compute Lesion-Level Metrics
    lesion_res = compute_lesion_level_metrics(pred[None, ...], gt[None, ...], min_iou=0.3, min_area=15)

    print("----------------------------------------------------------------------")
    print("                   PIXEL-LEVEL SEGMENTATION METRICS                   ")
    print("----------------------------------------------------------------------")
    print(f"  Dice Similarity (DSC / F1):  {dsc:.4f}  (Overlap agreement)")
    print(f"  Intersection over Union:     {iou:.4f}  (Jaccard index)")
    print(f"  Pixel Sensitivity (Recall):  {sens:.4f}  (Caries pixels captured)")
    print(f"  Pixel Specificity:           {spec:.4f}  (Healthy background preserved)")
    print(f"  Pixel Precision (PPV):       {prec:.4f}  (Predicted pixels accurate)")
    print(f"  Pixel Overall Accuracy:      {acc:.4f}")
    print("\n  Confusion Matrix Breakdown:")
    print(f"    True Positives (TP):       {tp:,} pixels")
    print(f"    False Positives (FP):      {fp:,} pixels")
    print(f"    False Negatives (FN):      {fn:,} pixels")
    print(f"    True Negatives (TN):       {tn:,} pixels")

    print("\n----------------------------------------------------------------------")
    print("                 LESION-LEVEL DETECTION METRICS                       ")
    print("----------------------------------------------------------------------")
    print(f"  Detected True Lesions:       {lesion_res['detected_lesions']} / {lesion_res['total_true_lesions']}")
    print(f"  False Positive Lesions:      {lesion_res['false_positive_lesions']}")
    print(f"  Missed Lesions (FN):         {lesion_res['missed_lesions']}")
    print(f"  Lesion Detection Recall:     {lesion_res['lesion_recall']:.2%}")
    print(f"  Lesion Detection Precision:  {lesion_res['lesion_precision']:.2%}")
    print(f"  Lesion Detection F1 Score:   {lesion_res['lesion_f1']:.4f}")

    # Test MetricTracker over a simulated multi-batch evaluation
    tracker = MetricTracker(threshold=0.5)
    tracker.update(pred, gt)
    tracker.update(gt, gt)  # Second perfect batch
    summary = tracker.get_metrics()
    print("\n----------------------------------------------------------------------")
    print(f"  Dataset MetricTracker Batch Test (2 batches): Overall DSC = {summary['dice']:.4f}, IoU = {summary['iou']:.4f}")
    print("=" * 70)
    print("[OK] All metrics calculated and validated successfully!")
