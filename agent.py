import time
import chess

from nnue_eval import evaluate_nnue

USE_LEARNED_EVAL = True

def evaluate_learned(board: chess.Board) -> int:
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    return evaluate_nnue(board)

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

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
killer_moves = [[None, None] for _ in range(64)]


def is_endgame(board: chess.Board) -> bool:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) +
              len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
    if queens == 0:
        return True
    if queens == 1 and minors <= 1:
        return True
    return False


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if USE_LEARNED_EVAL:
        return evaluate_nnue(board)

    endgame = is_endgame(board)
    score = 0

    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.piece_type == chess.KING:
            table = KING_ENDGAME_TABLE if endgame else KING_MIDDLEGAME_TABLE
        else:
            table = PST.get(piece.piece_type)

        if piece.color == chess.WHITE:
            pst_bonus = table[chess.square_mirror(square)] if table else 0
            score += val + pst_bonus
        else:
            pst_bonus = table[square] if table else 0
            score -= val + pst_bonus

    # Bishop pair bonus
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 50

    # Passed pawn bonus (helps convert endgames)
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        is_passed = True
        for opp_sq in board.pieces(chess.PAWN, chess.BLACK):
            if abs(chess.square_file(opp_sq) - file) <= 1 and chess.square_rank(opp_sq) > rank:
                is_passed = False
                break
        if is_passed:
            score += rank * rank * 3

    for sq in board.pieces(chess.PAWN, chess.BLACK):
        rank = 7 - chess.square_rank(sq)
        file = chess.square_file(sq)
        is_passed = True
        for opp_sq in board.pieces(chess.PAWN, chess.WHITE):
            if abs(chess.square_file(opp_sq) - file) <= 1 and (7 - chess.square_rank(opp_sq)) > rank:
                is_passed = False
                break
        if is_passed:
            score -= rank * rank * 3

    if board.turn == chess.WHITE:
        return score
    else:
        return -score


def quiescence(board, alpha, beta, start_time, time_limit):
    if time.time() - start_time > time_limit:
        raise TimeoutError()

    in_check = board.is_check()
    if not in_check:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

    for move in order_moves(board):
        if not in_check and not board.is_capture(move):
            continue  # in check: search ALL moves, not just captures
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
        finally:
            board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def order_moves(board: chess.Board, tt_move=None, ply=0):
    def score_move(move: chess.Move):
        if tt_move and move == tt_move:
            return 100000
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            return 50000 + (victim_val - attacker_val)
        if ply < 64:
            if move == killer_moves[ply][0]:
                return 40000
            if move == killer_moves[ply][1]:
                return 39000
        if board.gives_check(move):
            return 5000
        return 0

    return sorted(board.legal_moves, key=score_move, reverse=True)


# TT flag constants
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float, ply: int = 0):
    if time.time() - start_time > time_limit:
        raise TimeoutError()

    # Repetition and draw detection (only in non-root nodes)
    if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
        return 0, None

    if board.is_game_over():
        if board.is_checkmate():
            return -30000 + ply, None  # prefer faster checkmates
        return 0, None  # stalemate or other draw

    if depth == 0:
        return quiescence(board, alpha, beta, start_time, time_limit), None

    key = board._transposition_key()
    tt_entry = transposition_table.get(key)
    tt_move = None
    if tt_entry:
        stored_depth, stored_score, stored_move, stored_flag = tt_entry
        tt_move = stored_move
        if stored_depth >= depth:
            if stored_flag == EXACT:
                return stored_score, stored_move
            elif stored_flag == LOWERBOUND and stored_score >= beta:
                return stored_score, stored_move
            elif stored_flag == UPPERBOUND and stored_score <= alpha:
                return stored_score, stored_move

    # Null move pruning (disabled in endgame to avoid zugzwang)
    if depth >= 3 and not board.is_check() and not is_endgame(board):
        board.push(chess.Move.null())
        try:
            null_score, _ = alpha_beta(board, depth - 3, -beta, -beta + 1, start_time, time_limit, ply + 1)
            null_score = -null_score
        finally:
            board.pop()
        if null_score >= beta:
            return beta, None

    best_move = None
    best_score = -float('inf')
    original_alpha = alpha

    for move in order_moves(board, tt_move=tt_move, ply=ply):
        board.push(move)
        try:
            score, _ = alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit, ply + 1)
            score = -score
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            # Record killer move for quiet moves that cause cutoff
            if not board.is_capture(move) and ply < 64:
                killer_moves[ply][1] = killer_moves[ply][0]
                killer_moves[ply][0] = move
            break

    # Store with proper bound type
    if best_score <= original_alpha:
        flag = UPPERBOUND
    elif best_score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT
    transposition_table[key] = (depth, best_score, best_move, flag)

    return best_score, best_move


def get_move(fen: str, time_left_ms: int) -> str:
    global killer_moves
    if len(transposition_table) > 500000:
        transposition_table.clear()
    killer_moves = [[None, None] for _ in range(64)]

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    best_move = legal_moves[0]
    start_time = time.time()

    allocated_seconds = max(0.1, min(6.0, (time_left_ms / 1000.0) * 0.05))
    if time_left_ms < 20000:
        allocated_seconds = 0.05
    elif time_left_ms < 40000:
        allocated_seconds = 0.1

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