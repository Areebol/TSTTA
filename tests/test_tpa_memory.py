import unittest
import sys
import types

import torch

try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

from tta.tpa_memory import (
    AnchorRecord,
    DiverseOnlineAnchorBuffer,
    TPAOnlineMemory,
    TPAPrototypeAdapter,
)


def make_record(sample_id, query, error):
    return AnchorRecord(
        sample_id=sample_id,
        query=torch.tensor(query, dtype=torch.float32).view(1, -1),
        y_base=torch.zeros(2, 1),
        y_gt=torch.zeros(2, 1),
        adapter_delta=torch.zeros(2, 1),
        base_error=error,
    )


class DiverseAnchorBufferTest(unittest.TestCase):
    def test_nearest_pair_discards_easier_record(self):
        buffer = DiverseOnlineAnchorBuffer(capacity=2)
        buffer.add(make_record(0, [1.0, 0.0], 1.0))
        buffer.add(make_record(1, [0.999, 0.001], 2.0))
        removed = buffer.add(make_record(2, [-1.0, 0.0], 0.5))
        self.assertEqual(removed, 0)
        self.assertEqual({record.sample_id for record in buffer.records}, {1, 2})


class DistillationTest(unittest.TestCase):
    def setUp(self):
        self.adapter = TPAPrototypeAdapter(
            window_len=2,
            n_var=1,
            seq_len=4,
            n_static=2,
            n_online=1,
            feature_dim=2,
        )

    def test_mean_only_averages_sample_dimension(self):
        memory = TPAOnlineMemory(self.adapter, distill_mode="mean")
        query = torch.tensor([[[1.0, 0.0]], [[0.8, 0.6]]])
        y_base = torch.zeros(2, 2, 1)
        y_final = torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]])
        _, value, coherence = memory.distill_candidate(
            query, y_base, y_final
        )
        self.assertTrue(torch.allclose(value, torch.tensor([[3.0, 5.0]])))
        self.assertNotEqual(float(value[0, 0]), float(value[0, 1]))
        self.assertGreater(coherence, 0.0)

    def test_query_weighting_is_per_variable_across_samples(self):
        memory = TPAOnlineMemory(
            self.adapter, distill_mode="query_weighted"
        )
        query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        y_base = torch.zeros(2, 2, 1)
        y_final = torch.tensor([[[1.0], [2.0]], [[9.0], [10.0]]])
        key, value, _ = memory.distill_candidate(query, y_base, y_final)
        similarity = torch.einsum("mvd,vd->mv", query, key)
        weights = torch.softmax(self.adapter.temperature * similarity, dim=0)
        expected = torch.einsum(
            "mv,mhv->vh", weights, y_final - y_base
        )
        self.assertTrue(torch.allclose(value, expected))
        self.assertEqual(tuple(value.shape), (1, 2))

    def test_candidate_is_admitted_only_when_replay_improves(self):
        with torch.no_grad():
            self.adapter.static_keys.fill_(-1.0)
            self.adapter.static_values.zero_()
        memory = TPAOnlineMemory(self.adapter, distill_mode="mean")
        query = torch.tensor([[[1.0, 0.0]]])
        y_base = torch.zeros(1, 2, 1)
        y_gt = torch.ones(1, 2, 1)
        result = memory.process_delayed_batch(
            query=query,
            y_base=y_base,
            y_final=y_gt,
            y_gt=y_gt,
            adapter_delta=torch.zeros_like(y_gt),
        )
        self.assertTrue(result["accepted"])
        self.assertLess(
            result["best_replay_mse"], result["baseline_replay_mse"]
        )
        self.assertEqual(self.adapter.online_count, 1)

    def test_multivariate_input_selects_only_forecast_targets(self):
        adapter = TPAPrototypeAdapter(
            window_len=2,
            n_var=2,
            seq_len=4,
            n_static=2,
            n_online=1,
            feature_dim=2,
            target_start_idx=3,
        )
        adapter.online_mode = True
        x = torch.randn(3, 4, 7)
        y_base = torch.randn(3, 2, 2)
        y_final, details = adapter.forward_with_details(y_base, x)
        self.assertEqual(tuple(y_final.shape), (3, 2, 2))
        self.assertEqual(tuple(details["query"].shape), (3, 2, 2))
        self.assertEqual(tuple(details["frequency_delta"].shape), (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
