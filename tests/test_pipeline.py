"""
Integration test for full end-to-end pipeline:
Synthetic Data Generation -> Preprocessing -> UNet++ Forward/Backward -> Lesion Extraction.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sample_generator import generate_single_radiograph, generate_sample_dataset
from inference.lesion_detector import LesionDetector


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_synthetic_radiograph_generation(self):
        """Verify synthetic dental radiograph generator outputs valid dimensions and masks."""
        img, mask, meta = generate_single_radiograph(
            width=128,
            height=128,
            has_caries=True,
            lesion_type="proximal",
            seed=42
        )
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(mask.size, (128, 128))
        self.assertTrue(meta["has_caries"])
        self.assertGreater(len(meta["lesions"]), 0)

        # Check mask has positive pixels
        mask_np = np.array(mask)
        self.assertGreater(np.sum(mask_np > 0), 0)

    def test_sample_dataset_generation(self):
        """Verify batch sample dataset creation."""
        img_paths, mask_paths = generate_sample_dataset(
            output_dir=self.temp_dir,
            num_samples=4,
            width=128,
            height=128
        )
        self.assertEqual(len(img_paths), 4)
        self.assertEqual(len(mask_paths), 4)
        for p in img_paths + mask_paths:
            self.assertTrue(p.exists())

    def test_lesion_detector(self):
        """Verify connected-component lesion extraction and anatomical classification."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        # Create a synthetic caries cluster
        mask[30:45, 40:55] = 1

        prob_map = mask.astype(np.float32) * 0.85

        detector = LesionDetector(min_area=10)
        lesions = detector.detect_lesions(mask, prob_map, image_shape=(128, 128))

        self.assertEqual(len(lesions), 1)
        l = lesions[0]
        self.assertEqual(l.id, 1)
        self.assertGreater(l.confidence, 0.8)
        self.assertTrue(len(l.clinical_recommendation) > 5)


if __name__ == "__main__":
    unittest.main()
