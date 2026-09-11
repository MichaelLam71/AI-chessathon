import chess
from nnue_eval import evaluate_nnue

print("=" * 60)
print("NNUE EVALUATION DIAGNOSTIC")
print("=" * 60)

# 1. Starting position (should be ~0)
board = chess.Board()
print(f"\n--- BASIC POSITIONS ---")
print(f"Starting position:        {evaluate_nnue(board):>6}  (expect ~0)")

board = chess.Board()
board.push_san("e4")
print(f"After 1. e4 (black STM):  {evaluate_nnue(board):>6}  (expect ~-20)")

board = chess.Board()
board.push_san("e4")
board.push_san("e5")
print(f"After 1. e4 e5 (white STM): {evaluate_nnue(board):>6}  (expect ~0)")

# 2. Material imbalance symmetry tests
print(f"\n--- MATERIAL SYMMETRY (same imbalance, different colour) ---")

# Queen
w_up_q = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
b_up_q = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
w_score = evaluate_nnue(w_up_q)
b_score = evaluate_nnue(b_up_q)
print(f"White up queen (white STM):  {w_score:>6}  (expect ~+900)")
print(f"Black up queen (white STM):  {b_score:>6}  (expect ~-900)")
print(f"  Asymmetry: {abs(w_score) - abs(b_score):>6}  (expect ~0)")

# Rook
w_up_r = chess.Board("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1")
b_up_r = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
w_score = evaluate_nnue(w_up_r)
b_score = evaluate_nnue(b_up_r)
print(f"White up rook (white STM):   {w_score:>6}  (expect ~+500)")
print(f"Black up rook (white STM):   {b_score:>6}  (expect ~-500)")
print(f"  Asymmetry: {abs(w_score) - abs(b_score):>6}  (expect ~0)")

# Knight
w_up_n = chess.Board("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
b_up_n = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w Qkq - 0 1")
w_score = evaluate_nnue(w_up_n)
b_score = evaluate_nnue(b_up_n)
print(f"White up knight (white STM): {w_score:>6}  (expect ~+320)")
print(f"Black up knight (white STM): {b_score:>6}  (expect ~-320)")
print(f"  Asymmetry: {abs(w_score) - abs(b_score):>6}  (expect ~0)")

# Pawn
w_up_p = chess.Board("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
b_up_p = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
w_score = evaluate_nnue(w_up_p)
b_score = evaluate_nnue(b_up_p)
print(f"White up pawn (white STM):   {w_score:>6}  (expect ~+100)")
print(f"Black up pawn (white STM):   {b_score:>6}  (expect ~-100)")
print(f"  Asymmetry: {abs(w_score) - abs(b_score):>6}  (expect ~0)")

# 3. Side-to-move consistency
print(f"\n--- SIDE TO MOVE CONSISTENCY ---")
w_up_q_wtm = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
w_up_q_btm = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
w_score = evaluate_nnue(w_up_q_wtm)
b_score = evaluate_nnue(w_up_q_btm)
print(f"White up queen, white STM:   {w_score:>6}  (expect large positive)")
print(f"White up queen, black STM:   {b_score:>6}  (expect large negative)")
print(f"  Sum (should be ~0):        {w_score + b_score:>6}")

# 4. Positional understanding
print(f"\n--- POSITIONAL UNDERSTANDING ---")

# Developed vs undeveloped
developed = chess.Board("r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
undeveloped = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
print(f"Italian game (developed):    {evaluate_nnue(developed):>6}  (expect slight positive)")
print(f"After 1.e4 (undeveloped):    {evaluate_nnue(undeveloped):>6}  (expect ~0)")

# Castled king safety
castled = chess.Board("rnbq1rk1/ppppbppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 5")
uncastled = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
print(f"Both castled:                {evaluate_nnue(castled):>6}")
print(f"Neither castled:             {evaluate_nnue(uncastled):>6}")

# 5. Endgame understanding
print(f"\n--- ENDGAME ---")
kp_winning = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
kp_losing = chess.Board("4K3/4p3/8/8/8/4k3/8/8 b - - 0 1")
print(f"KP vs K (white winning):     {evaluate_nnue(kp_winning):>6}  (expect positive)")
print(f"KP vs K (black winning):     {evaluate_nnue(kp_losing):>6}  (expect positive for black)")

# 6. Ordering test: does eval rank positions correctly?
print(f"\n--- RANKING TEST (should be descending) ---")
positions = [
    ("White up queen",   "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("White up rook",    "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1"),
    ("White up knight",  "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("White up pawn",    "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Equal",            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Black up pawn",    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1"),
    ("Black up knight",  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w Qkq - 0 1"),
    ("Black up rook",    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"),
    ("Black up queen",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
]

scores = [(name, evaluate_nnue(chess.Board(fen))) for name, fen in positions]
for name, score in scores:
    bar = "+" * max(0, score // 20) + "-" * max(0, -score // 20)
    print(f"  {name:20s}: {score:>6}  {bar}")

correctly_ordered = all(scores[i][1] >= scores[i+1][1] for i in range(len(scores)-1))
print(f"\n  Correctly ranked: {'YES' if correctly_ordered else 'NO'}")

print(f"\n{'=' * 60}")