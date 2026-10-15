"""
Unit tests for UNet++ (Nested U-Net) Architecture.
"""

from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet_plus_plus import UNetPlusPlus, build_unet_plus_plus


class TestUNetPlusPlus(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_unet_plus_plus(
            in_channels=1,
            num_classes=1,
            deep_supervision=True,
            use_attention=True,
            nb_filter=[16, 32, 64, 128, 256]
        ).to(self.device)

    def test_train_forward_deep_supervision(self):
        """During training with deep supervision, model must output 4 feature maps."""
        self.model.train()
        x = torch.randn(2, 1, 128, 128, device=self.device)
        outputs = self.model(x)

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 4)
        for i, out in enumerate(outputs):
            self.assertEqual(out.shape, (2, 1, 128, 128), f"Output {i} shape mismatch")

    def test_eval_forward_single_output(self):
        """In evaluation mode, model should return single final resolution mask."""
        self.model.eval()
        x = torch.randn(2, 1, 128, 128, device=self.device)
        out = self.model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_pruning_levels(self):
        """Verify dynamic sub-network pruning (L1 to L4)."""
        self.model.eval()
        x = torch.randn(1, 1, 128, 128, device=self.device)

        for level in [1, 2, 3, 4]:
            self.model.set_pruning_level(level)
            out = self.model(x)
            self.assertEqual(out.shape, (1, 1, 128, 128), f"Pruning L{level} failed")

        self.model.set_pruning_level(None)

    def test_backward_pass(self):
        """Verify backward gradients propagate through all nested skip pathways."""
        self.model.train()
        x = torch.randn(1, 1, 64, 64, device=self.device)
        outputs = self.model(x)
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")


if __name__ == "__main__":
    unittest.main()
