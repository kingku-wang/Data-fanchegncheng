from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import torch  # noqa: E402

from adafractcn import get_adafractcn  # noqa: E402
from baselines import get_model  # noqa: E402
from param_budget import manuscript_receptive_field, usable_receptive_field  # noqa: E402


class RepositorySmokeTests(unittest.TestCase):
    def test_receptive_field_contract(self):
        self.assertEqual(manuscript_receptive_field("TCN")[0], 61)
        self.assertEqual(manuscript_receptive_field("TCN-RFmatched")[0], 509)
        self.assertEqual(manuscript_receptive_field("AdaFracTCN")[0], 1006)
        self.assertEqual(usable_receptive_field("AdaFracTCN", 256)[0], 256)

    def test_length_256_forward_paths(self):
        x = torch.randn(2, 256, 1)
        for name in ["TCN", "Transformer", "Informer"]:
            model = get_model(name, input_len=256, horizon=1, mode="standard")
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 1), name)

        ada = get_adafractcn("AdaFracTCN", input_len=256, horizon=1, mode="standard")
        y = ada(x)
        self.assertEqual(tuple(y.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
