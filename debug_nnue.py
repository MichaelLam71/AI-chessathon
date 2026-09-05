import chess
from nnue_eval import evaluate_nnue

board = chess.Board()
print(f"Starting position: {evaluate_nnue(board)}")

board = chess.Board()
board.push_san("e4")
print(f"After 1. e4: {evaluate_nnue(board)}")

board = chess.Board()
board.push_san("e4")
board.push_san("e5")
print(f"After 1. e4 e5: {evaluate_nnue(board)}")

# White up a queen
board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(f"White up a queen: {evaluate_nnue(board)}")

# Black up a queen
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
print(f"Black up a queen: {evaluate_nnue(board)}")

# White up a queen, black to move
board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
print(f"White up queen, black to move: {evaluate_nnue(board)}")