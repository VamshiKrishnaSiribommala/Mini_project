import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest
import torch
import numpy as np
from src.DataHandler.tensorDICOM import DICOMImagePreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preproc = DICOMImagePreprocessor()
        self.sample_dicom = Path("data/openi/dicom/1202_IM-0136-1001.dcm")

    def test_dicom_loading(self):
        if not self.sample_dicom.exists():
            self.skipTest(f"Sample DICOM not found at {self.sample_dicom}")
        
        with open(self.sample_dicom, "rb") as f:
            dcm_bytes = f.read()
        tensor = self.preproc(dcm_bytes)
        self.assertIsInstance(tensor, torch.Tensor)
        # Verify shape: [3, 224, 224]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertFalse(torch.isnan(tensor).any())

    def test_synthetic_image_fallback(self):
        synthetic = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(synthetic)
        tensor = self.preproc.transform(pil_img).unsqueeze(0)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

if __name__ == '__main__':
    unittest.main()
