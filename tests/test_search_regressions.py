"""User-run regression checks; these are not a playing-strength benchmark."""
import time
import unittest
from unittest.mock import patch

import chess
import numpy as np

import agent


def material(board: chess.Board) -> int:
    score = sum(
        agent.PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
        for piece in board.piece_map().values()
    )
    return score


def reference(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return material(board)
    moves = list(board.legal_moves)
    if not moves:
        return -agent.MATE if board.is_check() else 0
    scores = []
    for move in moves:
        board.push(move)
        try:
            scores.append(-reference(board, depth - 1))
        finally:
            board.pop()
    return max(scores)


class SearchRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        agent.transposition_table.clear()
        agent.counter_moves.clear()
        agent.killer_moves = [[None, None] for _ in range(agent.MAX_PLY)]
        agent.history_table = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        agent.node_count = 0
        agent.search_start_time = time.time()
        agent.search_time_limit = 60.0
        agent.reset_search_stats()

    def test_legal_exchange_and_pinned_recapture(self) -> None:
        for fen, expected in (
            ("7k/8/4p3/3p4/2B5/8/8/K7 w - - 0 1", -230),
            ("4k3/8/4p3/3p4/2B5/8/8/K3R3 w - - 0 1", 100),
        ):
            board = chess.Board(fen)
            move = chess.Move.from_uci("c4d5")
            self.assertIn(move, board.legal_moves)
            self.assertEqual(agent.exchange_value(board, move), expected)
            self.assertEqual(board.fen(), fen)
            self.assertEqual(board.move_stack, [])

    def test_exchange_timeout_restores_board(self) -> None:
        board = chess.Board("7k/8/4p3/3p4/2B5/8/8/K7 w - - 0 1")
        original = board.fen()
        agent.search_start_time = 0.0
        agent.search_time_limit = 0.0
        with self.assertRaises(TimeoutError):
            agent.exchange_value(board, chess.Move.from_uci("c4d5"))
        self.assertEqual(board.fen(), original)
        self.assertEqual(board.move_stack, [])

    def test_en_passant_is_kept(self) -> None:
        board = chess.Board("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1")
        move = chess.Move.from_uci("e5d6")
        original = board.fen()
        self.assertTrue(board.is_legal(move))
        self.assertEqual(agent.captured_value(board, move), 100)
        self.assertEqual(agent.exchange_value(board, move), 0)
        self.assertEqual(board.fen(), original)

    def test_quiet_promotion_is_searched(self) -> None:
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        original = board.fen()
        with patch.object(agent, "USE_LEARNED_EVAL", False), patch.object(agent, "evaluate", material):
            self.assertGreaterEqual(agent.quiescence(board, -agent.INF, agent.INF), 900)
        self.assertEqual(board.fen(), original)
        self.assertEqual(board.move_stack, [])

    def test_qsearch_terminal_scores(self) -> None:
        mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(agent.quiescence(mate, -agent.INF, agent.INF, 5), -agent.MATE + 5)
        self.assertEqual(agent.quiescence(stalemate, -agent.INF, agent.INF), 0)

    def test_pvs_matches_full_width_at_depth_two(self) -> None:
        # At depth two neither LMR nor null pruning is eligible. Replace the leaf
        # evaluator so this checks PVS window/re-search logic independently of NNUE.
        for fen in (chess.STARTING_FEN,
                    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"):
            board = chess.Board(fen)
            expected = reference(board, 2)
            def leaf(position, alpha, beta, ply=0, allow_draw=True):
                return material(position)
            with patch.object(agent, "USE_LEARNED_EVAL", False), patch.object(agent, "quiescence", leaf):
                score, move = agent.alpha_beta(board, 2, -agent.INF, agent.INF)
            self.assertEqual(score, expected)
            self.assertIn(move, board.legal_moves)
            self.assertEqual(board.fen(), fen)
            self.assertEqual(board.move_stack, [])

    def test_tt_replacement_and_mate_normalization(self) -> None:
        move = chess.Move.from_uci("e2e4")
        agent.store_tt(7, 5, agent.MATE - 8, move, agent.EXACT, 3)
        agent.store_tt(7, 2, 0, move, agent.UPPERBOUND, 0)
        self.assertEqual(agent.transposition_table[7][1], 5)
        self.assertEqual(agent.transposition_table[7][2], agent.MATE - 5)
        agent.tt_generation += 1
        collision = 7 + agent.TT_SIZE
        agent.store_tt(collision, 2, 0, move, agent.EXACT, 0)
        self.assertEqual(len(agent.transposition_table), 1)
        self.assertEqual(agent.transposition_table[7][0], collision)

    def test_nested_timeout_restores_nnue_and_board(self) -> None:
        board = chess.Board()
        agent.accumulator.init_from_board(board)
        before_w = agent.accumulator.w_acc.copy()
        before_b = agent.accumulator.b_acc.copy()
        before_fen = board.fen()
        calls = 0
        def deadline():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise TimeoutError()
        with patch.object(agent, "check_time", deadline):
            with self.assertRaises(TimeoutError):
                agent.alpha_beta(board, 3, -agent.INF, agent.INF)
        self.assertEqual(board.fen(), before_fen)
        self.assertEqual(board.move_stack, [])
        self.assertEqual(agent.accumulator.stack, [])
        np.testing.assert_array_equal(agent.accumulator.w_acc, before_w)
        np.testing.assert_array_equal(agent.accumulator.b_acc, before_b)


if __name__ == "__main__":
    unittest.main()
