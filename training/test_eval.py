import chess
from integrate_example import evaluate_learned

print("start position:", evaluate_learned(chess.Board()))

fen_up_a_queen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print("white up a queen:", evaluate_learned(chess.Board(fen_up_a_queen)))