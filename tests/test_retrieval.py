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
from src.Retrieval.reranker import Reranker
from src.Retrieval.retrieval import make_retrieval_engine

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.features_path = Path("embeddings/train_joint_embeddings.npy")
        self.ids_path = Path("embeddings/train_ids.json")
        self.graph_path = Path("featureDBs/train_joint_embeddings_link_graph.pkl")

    def test_reranker_initialization(self):
        reranker = Reranker(alpha=0.60, beta=0.25, gamma=0.15, preload_record_kg=False)
        total = reranker.alpha + reranker.beta + reranker.gamma
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_retriever_query(self):
        if not self.features_path.exists() or not self.ids_path.exists():
            self.skipTest("Feature database files not found")

        retriever = make_retrieval_engine(
            features_path=str(self.features_path),
            ids_path=str(self.ids_path),
            graph_path=str(self.graph_path) if self.graph_path.exists() else None,
            method="dls"
        )
        
        # Query with random normalized vector of dimension 1024
        query_vec = np.random.randn(1024).astype(np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        
        ids, scores = retriever.retrieve(query_vec, K=5)
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), 5)
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 5)

if __name__ == '__main__':
    unittest.main()
