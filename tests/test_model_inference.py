import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest
import torch
from src.Model.fusion import CrossModalFusion

class TestModelInference(unittest.TestCase):
    def test_cross_modal_fusion_forward(self):
        batch_size = 2
        img_dim = 1024
        txt_dim = 768
        joint_dim = 256
        num_heads = 4
        
        fusion = CrossModalFusion(
            img_dim=img_dim,
            txt_dim=txt_dim,
            joint_dim=joint_dim,
            num_heads=num_heads
        )
        
        img_global = torch.randn(batch_size, img_dim)
        img_patch = torch.randn(batch_size, 49, img_dim)
        txt_feats = torch.randn(batch_size, 16, txt_dim)
        
        fused, attn = fusion(img_global, img_patch, txt_feats)
        
        # Default returns sequence of tokens: [cls_tok, patch_toks, txt_tok] -> (B, 1 + 49 + 1, joint_dim)
        self.assertEqual(fused.shape, (batch_size, 1 + 49 + 1, joint_dim))
        self.assertFalse(torch.isnan(fused).any())

if __name__ == '__main__':
    unittest.main()
