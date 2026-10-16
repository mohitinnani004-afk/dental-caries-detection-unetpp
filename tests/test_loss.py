"""
Unit tests for dental caries loss functions and metrics.
"""

from pathlib import Path
import sys
import unittest
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TORCH_AVAILABLE:
    from models.losses import DiceLoss, FocalLoss, BCEDiceLoss, DeepSupervisionLoss
    from models.metrics import MetricTracker, dice_coefficient, iou_score


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for neural net loss tests")
class TestLossesAndMetrics(unittest.TestCase):
    def setUp(self):
        self.preds = torch.randn(4, 1, 64, 64)
        self.targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    def test_dice_loss(self):
        criterion = DiceLoss()
        loss = criterion(self.preds, self.targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0 + 1e-4)

    def test_focal_loss(self):
        criterion = FocalLoss()
        loss = criterion(self.preds, self.targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_bce_dice_loss(self):
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        loss = criterion(self.preds, self.targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_deep_supervision_loss(self):
        criterion = DeepSupervisionLoss(weights=[0.1, 0.2, 0.3, 0.4])
        multi_preds = [torch.randn(4, 1, 64, 64) for _ in range(4)]
        loss = criterion(multi_preds, self.targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_metric_tracker(self):
        tracker = MetricTracker(threshold=0.5)
        tracker.update(self.preds, self.targets)
        metrics = tracker.get_metrics()

        for key in ["dice", "iou", "sensitivity", "specificity", "precision", "accuracy"]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)


if __name__ == "__main__":
    unittest.main()
