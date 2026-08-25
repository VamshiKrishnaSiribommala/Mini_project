import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest

class TestImports(unittest.TestCase):
    def test_core_dependencies(self):
        import torch
        import torchvision
        import timm
        import transformers
        import pydicom
        import captum
        import sklearn
        import networkx
        import yaml
        self.assertTrue(hasattr(torch, '__version__'))
        self.assertTrue(hasattr(pydicom, '__version__'))

    def test_lungxpert_modules(self):
        from src.Helpers.config import Config
        from src.DataHandler.tensorDICOM import DICOMImagePreprocessor
        from src.Model.fusion import CrossModalFusion
        from src.Retrieval.reranker import Reranker
        from src.KnowledgeGraph.compgcn_conv import CompGCNConv
        self.assertIsNotNone(Config)
        self.assertIsNotNone(DICOMImagePreprocessor)
        self.assertIsNotNone(CrossModalFusion)
        self.assertIsNotNone(Reranker)
        self.assertIsNotNone(CompGCNConv)

if __name__ == '__main__':
    unittest.main()
