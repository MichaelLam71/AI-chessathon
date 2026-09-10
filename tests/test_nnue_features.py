import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

import nnue_eval
from train_nnue import BASE_FEATURE_DIM, KING_BUCKET_FEATURE_DIM, NNUE, fen_to_features


class NNUEFeatureTests(unittest.TestCase):
    def test_king_bucket_keeps_base_piece_square_features(self) -> None:
        fen = "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 1"
        simple_w, simple_b = fen_to_features(fen)
        bucket_w, bucket_b = fen_to_features(fen, "king-bucket")
        self.assertEqual([index % BASE_FEATURE_DIM for index in bucket_w], simple_w)
        self.assertEqual([index % BASE_FEATURE_DIM for index in bucket_b], simple_b)
        self.assertEqual(len({index // BASE_FEATURE_DIM for index in bucket_w}), 1)
        self.assertEqual(len({index // BASE_FEATURE_DIM for index in bucket_b}), 1)

    def test_sparse_accumulation_matches_dense_one_hot(self) -> None:
        model = NNUE()
        indices, _ = fen_to_features(chess.STARTING_FEN)
        sparse = torch.full((1, 32), -1, dtype=torch.long)
        sparse[0, :len(indices)] = torch.tensor(indices)
        dense = torch.zeros((1, BASE_FEATURE_DIM))
        dense[0, indices] = 1.0
        expected = dense @ model.ft_weight.T + model.ft_bias
        torch.testing.assert_close(model.accumulate(sparse), expected)

    def test_bucket_crossing_rebuilds_incremental_accumulator(self) -> None:
        weights = np.random.default_rng(19).standard_normal(
            (KING_BUCKET_FEATURE_DIM, len(nnue_eval.FT_BIAS)), dtype=np.float32
        )
        board = chess.Board()
        moves = (
            "e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1", "f8e7",
            "d2d3", "e8g8",
        )
        with (patch.object(nnue_eval, "FT_WEIGHT", weights),
              patch.object(nnue_eval, "KING_BUCKETED", True)):
            accumulator = nnue_eval.NNUEAccumulator()
            accumulator.init_from_board(board)
            for uci in moves:
                move = board.parse_uci(uci)
                accumulator.push(board, move)
                board.push(move)
                rebuilt = nnue_eval.NNUEAccumulator()
                rebuilt.init_from_board(board)
                np.testing.assert_allclose(accumulator.w_acc, rebuilt.w_acc, atol=1e-5)
                np.testing.assert_allclose(accumulator.b_acc, rebuilt.b_acc, atol=1e-5)
                self.assertEqual(accumulator.w_bucket, rebuilt.w_bucket)
                self.assertEqual(accumulator.b_bucket, rebuilt.b_bucket)


if __name__ == "__main__":
    unittest.main()
