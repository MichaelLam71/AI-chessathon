import time
import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables (from white's perspective, a1=index 0, h8=index 63)
# Values represent positional bonuses/penalties in centipawns

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MIDDLEGAME_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}

transposition_table = {}


def is_endgame(board: chess.Board) -> bool:
    """Detect endgame: no queens or queen with at most one minor piece per side."""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) +
              len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
    if queens == 0:
        return True
    if queens == 1 and minors <= 1:
        return True
    return False


def evaluate(board: chess.Board) -> int:
    """
    Evaluates board position relative to board.turn 
    Positive = side to move is winning.
    Negative = side to move is losing.
    """
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    endgame = is_endgame(board)
    score = 0

    for square, piece in board.piece_map().items():
        # Material value
        val = PIECE_VALUES.get(piece.piece_type, 0)

        # Piece-square table bonus
        if piece.piece_type == chess.KING:
            table = KING_ENDGAME_TABLE if endgame else KING_MIDDLEGAME_TABLE
        else:
            table = PST.get(piece.piece_type)

        if piece.color == chess.WHITE:
            # Tables are from white's perspective, rank 8 at top (index 0)
            # python-chess: a1=0, h8=63. Mirror vertically for white.
            pst_bonus = table[chess.square_mirror(square)] if table else 0
            score += val + pst_bonus
        else:
            # For black, use the square directly (already mirrored perspective)
            pst_bonus = table[square] if table else 0
            score -= val + pst_bonus

    # Return relative to side to move
    if board.turn == chess.WHITE:
        return score
    else:
        return -score

def quiescence(board, alpha, beta, start_time, time_limit):
    if time.time() - start_time > time_limit:
        raise TimeoutError()
    
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    
    for move in order_moves(board):
        if not board.is_capture(move):
            continue  # only search captures
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
        finally:
            board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    
    return alpha


def order_moves(board: chess.Board):
    """
    Orders legal moves so captures and checks are evaluated first.
    Prunes the search tree significantly via Alpha-Beta pruning.
    """
    def score_move(move: chess.Move):
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            # Prioritize taking high-value pieces with low-value pieces (MVV-LVA)
            return 10000 + (victim_val - attacker_val)
        if board.gives_check(move):
            return 5000
        return 0

    return sorted(board.legal_moves, key=score_move, reverse=True)


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float):
    """Recursive Alpha-Beta search using Negamax structure."""

    key = (board.fen(), depth)
    if key in transposition_table:
        return transposition_table[key]
    if time.time() - start_time > time_limit:
        raise TimeoutError()
        
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit), None

    best_move = None
    best_score = -float('inf')

    for move in order_moves(board):
        board.push(move)
        try:
            score, _ = alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit)
            score = -score
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            break  # Beta cutoff
    transposition_table[key] = (best_score, best_move)
    return best_score, best_move

# 3. Required Competition Entrypoint
def get_move(fen: str, time_left_ms: int) -> str:
    transposition_table.clear()
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    # Default fallback move in case time expires immediately
    best_move = legal_moves[0]
    start_time = time.time()

    # Dynamic time allocation (Time trouble)
    allocated_seconds = max(0.1, min(4.0, (time_left_ms / 1000.0) * 0.03))
    if time_left_ms < 20000:  # under 20 seconds
        allocated_seconds = 0.05  # 50ms per move, basically instant
    elif time_left_ms < 40000:  # under 40 seconds
        allocated_seconds = 0.1

    # Iterative Deepening
    depth = 1
    while depth <= 20:
        try:
            _, move = alpha_beta(
                board=board,
                depth=depth,
                alpha=-float('inf'),
                beta=float('inf'),
                start_time=start_time,
                time_limit=allocated_seconds
            )
            if move:
                best_move = move
            depth += 1
        except TimeoutError:
            break

    return best_move.uci()

