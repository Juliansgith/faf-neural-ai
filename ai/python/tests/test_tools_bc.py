"""Small regressions for tools cloning masks, validation coverage, and cache provenance."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import FIRE_STATES  # noqa: E402
from fa_train import tools_bc  # noqa: E402
from fa_train.tools_trainer import (ToolSegment, ToolsTrainer, normalized_distillation_loss)  # noqa: E402

V_INDEX = tools_bc.V_INDEX


def tiny_observation(n_units: int = 3) -> dict:
    n_verbs = len(V_INDEX)
    return {
        "ids": [f"u{i}" for i in range(n_units)],
        "cand_ids": [], "cand_kind": np.zeros(0, np.int8), "enh_names": [],
        "can_build": np.zeros((0, 2), np.int64), "can_enh": np.zeros((0, 2), np.int64),
        "mask_verb": np.ones(n_verbs, bool), "mask_subject": np.ones((n_verbs, n_units), bool),
    }


def diagnostics(before=0, after=0, overflow=0):
    return {"source": "recording_bc", "before_filter": before, "after_filter": after,
            "overflow_past_k": overflow, "kept_for_labels": min(after, tools_bc.K), "rejections": {},
            "per_verb": {}}


class CommandFilteringTest(unittest.TestCase):
    def test_invalid_single_subject_command_does_not_reserve_subject(self):
        x = tiny_observation()
        used = set()
        invalid = {"verb": "fire_state", "subject": ["u0"], "option": len(FIRE_STATES)}
        valid = {"verb": "fire_state", "subject": ["u0"], "option": 0}
        self.assertFalse(tools_bc.command_allowed(invalid, x, tools_bc.lookups(x), used))
        self.assertTrue(tools_bc.command_allowed(valid, x, tools_bc.lookups(x), used))
        accepted, diag = tools_bc.filter_commands([invalid, valid], x)
        self.assertEqual(accepted, [valid])
        self.assertEqual(diag["rejections"], {"option_invalid": 1})
        self.assertEqual((diag["before_filter"], diag["after_filter"], diag["overflow_past_k"]), (2, 1, 0))

    def test_diagnostics_separate_valid_overflow_from_rejection(self):
        x = tiny_observation(tools_bc.K + 2)
        commands = [{"verb": "fire_state", "subject": [uid], "option": 0} for uid in x["ids"]]
        accepted, diag = tools_bc.filter_commands(commands, x)
        self.assertEqual(len(accepted), tools_bc.K)
        self.assertEqual(diag["before_filter"], tools_bc.K + 2)
        self.assertEqual(diag["after_filter"], tools_bc.K + 2)
        self.assertEqual(diag["overflow_past_k"], 2)
        self.assertEqual(diag["kept_for_labels"], tools_bc.K)

    def test_empty_and_invalid_only_teacher_turns_have_different_masks(self):
        self.assertTrue(tools_bc.should_supervise_commands([], []))
        self.assertFalse(tools_bc.should_supervise_commands([{"verb": "bad"}], []))
        self.assertTrue(tools_bc.should_supervise_commands([{"verb": "bad"}, {"verb": "fire_state"}],
                                                           [{"verb": "fire_state"}]))

    def test_online_distillation_masks_invalid_only_but_keeps_intentional_end(self):
        trainer = object.__new__(ToolsTrainer)
        trainer.device = torch.device("cpu")
        x = tiny_observation(1)
        invalid = {"verb": "fire_state", "subject": ["u0"], "option": len(FIRE_STATES)}
        _, has = trainer._teacher_actions([[invalid], [], None], [x, x, x], T=1)
        self.assertEqual(has.tolist(), [0.0, 1.0, 0.0])


class DistillationNormalizationTest(unittest.TestCase):
    def test_invalid_teacher_row_does_not_dilute_valid_empty_end_loss(self):
        trainer = object.__new__(ToolsTrainer)
        x = tiny_observation(1)
        invalid = {"verb": "fire_state", "subject": ["u0"], "option": len(FIRE_STATES)}
        seg = ToolSegment((None, None), first_is_start=True)
        seg.obs = [x, x]
        seg.teacher = [[invalid], []]  # invalid-only row, then a deliberate END decision
        seg.rewards = [0.0, 0.0]

        rows, denominator = trainer._prepare_teacher_rows([seg], L=2)
        accepted, supervised = rows[id(seg)]
        self.assertEqual(accepted, [[], []])
        self.assertEqual(supervised, [False, True])
        self.assertEqual(denominator, 1.0)

        bc = torch.tensor([100.0, 4.0], requires_grad=True)
        valid = torch.ones(2)
        loss = normalized_distillation_loss(bc, valid, torch.tensor(supervised), denominator)
        self.assertEqual(loss.detach().item(), 4.0)  # the old non-None denominator would halve this to 2.0
        loss.backward()
        self.assertEqual(bc.grad.tolist(), [0.0, 1.0])


class _ScaleNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))

    def initial_state(self, batch, device):
        return None

    def evaluate(self, b, lab, state, **kwargs):
        value = self.value.expand(1)
        return None, None, value, torch.zeros((1, 1)), value * 0 + 1


class ReturnScaleTest(unittest.TestCase):
    def test_scale_selection_and_invalid_checkpoint_values(self):
        self.assertEqual(tools_bc.select_return_scale(7.02), (7.02, "dataset"))
        self.assertEqual(tools_bc.select_return_scale(7.02, {}), (7.02, "legacy_checkpoint_dataset"))
        self.assertEqual(tools_bc.select_return_scale(7.02, {"ret_std": 74.22}),
                         (74.22, "init_checkpoint"))
        for invalid in (0, -1, float("nan"), float("inf"), None, "bad", True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "ret_std.*finite and positive"):
                tools_bc.select_return_scale(7.02, {"ret_std": invalid})

    def test_selected_scale_reaches_actual_value_loss_evaluation_and_checkpoint(self):
        games = mock.Mock()
        games.games = [{"returns": np.asarray([0.0, 14.04])} for _ in range(2)]
        games.lengths = np.asarray([2, 2])
        batch = ({"belief": torch.zeros((1, 1))}, {}, torch.tensor([True]),
                 torch.tensor([74.22]), torch.tensor([1.0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode, source in (("fresh", "dataset"), ("initialized", "init_checkpoint"),
                                 ("legacy", "legacy_checkpoint_dataset")):
                with self.subTest(mode=mode):
                    initial = None
                    if mode != "fresh":
                        initial = root / f"{mode}-initial.pt"
                        checkpoint = {"size": "M", "model": _ScaleNet().state_dict()}
                        if mode == "initialized":
                            checkpoint["ret_std"] = 74.22
                        torch.save(checkpoint, initial)
                    output = root / f"{mode}-output.pt"
                    with mock.patch.object(tools_bc, "ToolsNet", _ScaleNet), \
                            mock.patch.object(tools_bc, "parameter_count", return_value=1), \
                            mock.patch.object(tools_bc, "prefetch", return_value=iter([batch])), \
                            mock.patch.object(tools_bc, "evaluate_held_out", return_value={"policy_loss": 1.0}) as evaluate:
                        state = tools_bc.train(root, output, games=games, init_checkpoint=initial, steps=1,
                                               length=1, burn=0, batch_chunks=1, device="cpu", log=lambda _: None)
                    chosen = 74.22 if mode == "initialized" else 7.020001
                    self.assertAlmostEqual(state["dataset_ret_std"], 7.020001)
                    self.assertAlmostEqual(state["ret_std"], chosen)
                    self.assertEqual(state["ret_std_source"], source)
                    # Real train() auxiliary loss: value starts at zero; policy loss is exactly one.
                    self.assertAlmostEqual(state["history"][0]["train_loss"],
                                           1 + 0.5 * (74.22 / chosen) ** 2, places=4)
                    self.assertAlmostEqual(evaluate.call_args.args[5], chosen)
                    saved = torch.load(output, weights_only=False)
                    self.assertEqual(saved["ret_std"], state["ret_std"])
                    self.assertEqual(saved["dataset_ret_std"], state["dataset_ret_std"])
                    self.assertEqual(saved["ret_std_source"], source)


class _PlanGames:
    def __init__(self):
        self.lengths = np.asarray([18, 31, 100])
        self.games = [{"meta": np.array(json.dumps({"map": f"map{i}", "faction": i + 1,
                                                       "ai": "easy", "ai_faction": 4 - i})),
                       "_cache_manifest": {"source_name": f"game{i}.jsonl.gz"}}
                      for i in range(3)]


class _LossNet:
    def __init__(self):
        self.last_metrics = {}

    def eval(self):
        pass

    def train(self):
        pass

    def initial_state(self, batch, device):
        return None

    def evaluate(self, b, lab, state, seq_len, starts=None):
        loss = b["loss"]
        self.last_metrics = {"verb": (torch.ones_like(loss) * 0.5, torch.ones_like(loss))}
        return None, None, torch.zeros_like(loss), None, loss


class _EvalGames(_PlanGames):
    def batch(self, picks, length, device):
        losses, supervised, starts, returns = [], [], [], []
        for gi, start in picks:
            for t in range(start, start + length):
                is_supervised = (t % 2) == 0
                supervised.append(float(is_supervised))
                losses.append(2.0 if is_supervised else 100.0)
                starts.append(t == 0)
                returns.append(0.0)
        n = len(losses)
        return ({"loss": torch.tensor(losses, device=device)}, {}, torch.tensor(starts, device=device),
                torch.tensor(returns, device=device), torch.tensor(supervised, device=device))


class HeldOutTest(unittest.TestCase):
    def test_coverage_visits_every_game_and_phase_without_global_truncation(self):
        games = _PlanGames()
        picks, coverage = tools_bc.balanced_held_out_chunks(games, [0, 1, 2], length=12)
        self.assertEqual(coverage["games_covered"], 3)
        self.assertEqual({i for i, _ in picks}, {0, 1, 2})
        self.assertEqual(coverage["chunk_length"], 12)
        self.assertGreaterEqual(coverage["phases"]["opening"], 3)
        self.assertGreaterEqual(coverage["phases"]["middle"], 2)
        self.assertGreaterEqual(coverage["phases"]["late"], 2)
        self.assertEqual(coverage["by_game"][1]["map"], "map1")

    def test_policy_loss_ignores_invalid_only_rows_and_reports_coverage(self):
        metrics = tools_bc.evaluate_held_out(_LossNet(), _EvalGames(), [0, 1, 2], length=12, burn=0,
                                             ret_std=1.0, device="cpu", max_chunks=1)
        self.assertAlmostEqual(metrics["policy_loss"], 2.0)
        self.assertAlmostEqual(metrics["loss"], 2.0)
        self.assertGreater(metrics["supervised_decisions"], 0)
        self.assertEqual(metrics["held_out_coverage"]["games_covered"], 3)
        self.assertGreater(metrics["held_out_coverage"]["chunks"], 1)


class CacheProvenanceTest(unittest.TestCase):
    def test_explicit_files_provenance_invalidation_and_legacy_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a, raw_b, cache = root / "raw-a", root / "raw-b", root / "cache"
            raw_a.mkdir()
            raw_b.mkdir()
            cache.mkdir()
            for folder in (raw_a, raw_b):
                (folder / "blueprints.json.gz").write_bytes(b"blueprints-v1")
            a = raw_a / "same.jsonl.gz"
            a.write_bytes(b"recording-a-v1")
            b = raw_b / "same.jsonl.gz"
            b.write_bytes(b"recording-b-v1")
            unused = raw_a / "not-selected.jsonl.gz"
            unused.write_bytes(b"not selected")
            legacy = cache / "legacy.v3.npz"
            np.savez_compressed(legacy, returns=np.zeros(1, np.float32))
            legacy_before = legacy.read_bytes()

            def encode(path, _bps):
                d = diagnostics(before=1, after=1)
                return {"returns": np.asarray([0.0, 1.0], np.float32),
                        "bc_supervise": np.asarray([True, False]),
                        "command_diagnostics": np.array(json.dumps(d)),
                        "meta": np.array(json.dumps({"map": "SCMP_016", "faction": 1, "ai": "easy"}))}, \
                    {"steps": 2, "command_diagnostics": d}

            with mock.patch.object(tools_bc, "read_blueprints", return_value=[]), \
                    mock.patch.object(tools_bc, "_encoder_semantics_hash", return_value="encoder-v1"), \
                    mock.patch.object(tools_bc, "encode_game", side_effect=encode):
                first = tools_bc.build_cache(raw_a, cache, log=lambda _: None, files=[a])
                self.assertEqual(first["games"], 1)
                self.assertEqual(len(list(cache.glob("*.v3.npz"))), 2)  # new + legacy; unselected raw omitted
                a.write_bytes(b"recording-a-v2")
                tools_bc.build_cache(raw_a, cache, log=lambda _: None, files=[a])
                tools_bc.build_cache(raw_b, cache, log=lambda _: None, files=[b])
                loaded = tools_bc.Games(cache)
                current_paths = tools_bc.current_cache_paths(cache)

            self.assertEqual(len(loaded.games), 2)  # newest version per source, basename collision kept distinct
            self.assertEqual(legacy.read_bytes(), legacy_before)
            self.assertEqual(len(current_paths), 3)
            self.assertTrue(all(g["bc_supervise"].tolist() == [True, False] for g in loaded.games))


if __name__ == "__main__":
    unittest.main()
