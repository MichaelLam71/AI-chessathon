import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from prepare_public_nnue import (
    file_month,
    phase,
    position_key,
    quota,
    save_chunk,
    score_bin,
    side_to_move_cp,
)
from train_nnue import CP_OUTPUT_SCALE, ChessNNUEDataset, ChunkedChessNNUEDataset


class PublicNNUEDataTests(unittest.TestCase):
    def test_white_scores_are_converted_to_side_to_move(self) -> None:
        white = chess.Board()
        black = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        self.assertEqual(side_to_move_cp(white, 37), 37)
        self.assertEqual(side_to_move_cp(black, 37), -37)

    def test_quota_rounding_preserves_requested_size(self) -> None:
        for size in (1, 100, 8_000_000):
            selected = quota(size, {"a": 0.2, "b": 0.55, "c": 0.25})
            self.assertEqual(sum(selected.values()), size)
            self.assertEqual(len(selected), 3)

    def test_score_bins_cover_boundaries(self) -> None:
        self.assertEqual(score_bin(-800), "loss_400_800")
        self.assertEqual(score_bin(-50), "near_equal")
        self.assertEqual(score_bin(49), "near_equal")
        self.assertEqual(score_bin(800), "win_800_plus")

    def test_source_month_is_parsed_from_shard(self) -> None:
        self.assertEqual(file_month("some/path/standard_rated_2025_03.parquet"), "2025-03")

    def test_phase_and_feature_key_ignore_move_counters(self) -> None:
        early = chess.Board()
        same = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 17 42")
        ending = chess.Board("8/8/8/3k4/8/3K4/4P3/8 w - - 0 40")
        self.assertEqual(phase(early), "opening")
        self.assertEqual(phase(ending), "endgame")
        self.assertEqual(position_key(early), position_key(same))

    def test_prepared_chunk_is_compatible_with_cp_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "chunk_public_00000.npz",
                fens=np.array([chess.STARTING_FEN]),
                scores=np.array([1500], dtype=np.int16),
            )
            dataset = ChessNNUEDataset(directory, target="cp", feature_set="simple")
            _, _, _, target = dataset[0]
            self.assertAlmostEqual(float(target), 1500 / CP_OUTPUT_SCALE, places=5)

    def test_resumable_chunk_contains_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [(chess.STARTING_FEN, 20, "opening", "quiet", "near_equal")]
            save_chunk(output, 0, rows, {"accepted": 1})
            with np.load(output / "chunk_public_00000.npz") as data:
                self.assertIn('"accepted": 1', str(data["checkpoint_json"].item()))

    def test_chunked_trainer_is_deterministic_and_normalizes_cp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_public_00000.npz"
            np.savez(
                path,
                fens=np.array([chess.STARTING_FEN, chess.STARTING_FEN]),
                scores=np.array([-1500, 1500], dtype=np.int16),
            )
            dataset = ChunkedChessNNUEDataset(Path(directory), "cp", "simple", seed=19)
            first = [float(sample[3]) for sample in dataset]
            second = [float(sample[3]) for sample in dataset]
            self.assertEqual(first, second)
            self.assertEqual(sorted(round(value, 5) for value in first), [
                round(-1500 / CP_OUTPUT_SCALE, 5), round(1500 / CP_OUTPUT_SCALE, 5)
            ])


if __name__ == "__main__":
    unittest.main()
