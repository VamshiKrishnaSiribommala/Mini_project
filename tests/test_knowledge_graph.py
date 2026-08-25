import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest
import torch
from src.KnowledgeGraph.compgcn_conv import CompGCNConv, ccorr

class TestKnowledgeGraph(unittest.TestCase):
    def test_ccorr_fft_operation(self):
        a = torch.randn(2, 300)
        b = torch.randn(2, 300)
        out = ccorr(a, b)
        self.assertEqual(out.shape, (2, 300))
        self.assertFalse(torch.isnan(out).any())

    def test_compgcn_conv_shapes(self):
        in_channels = 300
        out_channels = 300
        num_relations = 9
        conv = CompGCNConv(in_channels, out_channels, num_relations)
        
        # 4 nodes, 3 directed edges
        x = torch.randn(4, in_channels)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
        rel_embed = torch.randn(num_relations, in_channels)
        
        out_x, out_rel = conv(x, edge_index, edge_type, rel_embed)
        self.assertEqual(out_x.shape, (4, out_channels))
        self.assertEqual(out_rel.shape, (num_relations, out_channels))

    def test_label_attention_weights(self):
        ckpt_path = Path("label attention model/label_attention_model.pt")
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")
            
        state = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = state.get("model_state", state)
        self.assertIn("attn.0.weight", state_dict)
        self.assertIn("attn.2.weight", state_dict)
        # Verify Eq. (11) dimensions: W1 in R^(256 x 300), w2 in R^(1 x 256)
        self.assertEqual(state_dict["attn.0.weight"].shape, torch.Size([256, 300]))
        self.assertEqual(state_dict["attn.2.weight"].shape, torch.Size([1, 256]))

if __name__ == '__main__':
    unittest.main()
