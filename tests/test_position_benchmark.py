"""Accounting tests using fabricated labels; no engine searches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_position_results import paired_statistics
from position_benchmark import percentile, score_result, summarize


def position(best_cp: int | None, chosen_cp: int | None,
             best_mate: int | None = None, chosen_mate: int | None = None) -> dict[str, Any]:
    def move(uci: str, cp: int | None, mate: int | None) -> dict[str, Any]:
        return {"move": uci, "cp": cp, "mate": mate,
                "score_cp": cp if cp is not None else (100000 - abs(mate or 0)) * (
                    1 if (mate or 0) > 0 else -1)}
    return {"fen": "8/8/8/p3kP2/P7/8/5K2/8 b - - 2 69",
            "best_move": "e5f5", "top3": ["e5f5"],
            "moves": [move("e5f5", best_cp, best_mate),
                      move("e5d4", chosen_cp, chosen_mate)]}


class BenchmarkAccountingTests(unittest.TestCase):
    def test_black_pov_cp_loss(self) -> None:
        result = {"status": "ok", "move": "e5d4"}
        score_result(position(-10, -250), result)
        self.assertEqual(result["ordinary_loss_cp"], 240)

    def test_mate_loss_excluded_from_cp(self) -> None:
        result = {"status": "ok", "move": "e5d4"}
        score_result(position(0, None, chosen_mate=-10), result)
        summary = summarize([result, {"status": "timeout"}], 100, 200)
        self.assertIsNone(summary["mean_loss_cp"])
        self.assertEqual(summary["allowed_losing_mate"], 1)
        self.assertEqual(summary["blunders"], 0)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["synthetic_mean_loss_cp"], 99990)

    def test_missed_mate_and_distance(self) -> None:
        result = {"status": "ok", "move": "e5d4"}
        score_result(position(None, 500, best_mate=4), result)
        self.assertTrue(result["missed_winning_mate"])
        score_result(position(None, None, best_mate=4, chosen_mate=8), result)
        self.assertFalse(result["missed_winning_mate"])
        self.assertTrue(result["mate_distance_worsened"])

    def test_wdl_draw_estimates(self) -> None:
        record = position(0, -250)
        record["moves"][0]["wdl"] = [0, 1000, 0]
        record["moves"][1]["wdl"] = [0, 100, 900]
        result = {"status": "ok", "move": "e5d4"}
        score_result(record, result)
        self.assertTrue(result["estimated_draw_to_loss"])
        self.assertFalse(result["estimated_win_to_draw"])

    def test_percentile_and_threshold_boundaries(self) -> None:
        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([10], 90), 10)
        self.assertEqual(percentile([0, 100], 95), 95)
        results = []
        for loss in [0, 100, 200]:
            result = {"status": "ok", "move": "e5d4"}
            score_result(position(0, -loss), result)
            results.append(result)
        summary = summarize(results, 100, 200)
        self.assertEqual(summary["large_mistakes"], 2)
        self.assertEqual(summary["blunders"], 1)
        self.assertEqual(summary["median_loss_cp"], 100)

    def test_paired_game_bootstrap_and_fingerprint_guard(self) -> None:
        metadata = dict.fromkeys(("agent_sha256", "nnue_eval_sha256", "weights_sha256",
                                  "labels_sha256", "budget_ms", "time_left_ms", "mistake_cp",
                                  "blunder_cp", "mate_cp"), "same")
        c = dict(metadata, report_schema_version=2, evaluator="classical", results=[
            {"fen": "a", "game_id": "game", "best_agreement": False, "ordinary_loss_cp": 100},
            {"fen": "b", "game_id": "game", "best_agreement": True, "ordinary_loss_cp": 20}])
        n = dict(metadata, report_schema_version=2, evaluator="nnue", results=[
            {"fen": "a", "game_id": "game", "best_agreement": True, "ordinary_loss_cp": 20},
            {"fen": "b", "game_id": "game", "best_agreement": True, "ordinary_loss_cp": 20}])
        paired = paired_statistics(c, n)
        self.assertEqual(paired["nnue_cp_loss_reduction_ci95"], [40, 40])
        self.assertEqual(paired["nnue_exact_agreement_advantage_ci95"], [0.5, 0.5])
        n["labels_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "labels_sha256"):
            paired_statistics(c, n)


if __name__ == "__main__":
    unittest.main()
