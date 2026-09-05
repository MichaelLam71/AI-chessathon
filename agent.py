import time
import chess

from nnue_eval import accumulator, evaluate_nnue, NNUEAccumulator

USE_LEARNED_EVAL = True

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
history_table = [[0] * 64 for _ in range(64)]
node_count = 0
search_start_time = 0.0
search_time_limit = 0.0


def check_time():
    global node_count
    node_count += 1
    if (node_count & 255) == 0:  # was 2047 — check roughly every 256 nodes instead
        if time.time() - search_start_time > search_time_limit:
            raise TimeoutError()


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
    if USE_LEARNED_EVAL:
        return accumulator.evaluate(board.turn)

    # Classical eval fallback
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

    # Passed pawn bonus
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


def quiescence(board, alpha, beta, ply=0):
    check_time()

    if ply > 20:
        return evaluate(board)

    in_check = board.is_check()
    if not in_check:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if not any(board.legal_moves):
            return -30000

    moves = board.legal_moves if in_check else board.generate_legal_captures()
    for move in order_moves(board, moves, ply=ply):
        if USE_LEARNED_EVAL:
            accumulator.push(board, move)
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1)
        finally:
            board.pop()
            if USE_LEARNED_EVAL:
                accumulator.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def order_moves(board: chess.Board, moves=None, tt_move=None, ply=0):
    if moves is None:
        moves = board.legal_moves
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
        return history_table[move.from_square][move.to_square]

    return sorted(moves, key=score_move, reverse=True)


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0):
    check_time()

    if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
        return 0, None

    if depth == 0:
        return quiescence(board, alpha, beta), None

    key = board._transposition_key()
    tt_entry = transposition_table.get(key)
    tt_move = None
    if tt_entry:
        stored_depth, stored_score, stored_move, stored_flag = tt_entry
        tt_move = stored_move

        eval_score = stored_score
        if eval_score > 20000:
            eval_score -= ply
        elif eval_score < -20000:
            eval_score += ply

        if stored_depth >= depth:
            if stored_flag == EXACT:
                return eval_score, stored_move
            elif stored_flag == LOWERBOUND and eval_score >= beta:
                return eval_score, stored_move
            elif stored_flag == UPPERBOUND and eval_score <= alpha:
                return eval_score, stored_move

    # Null move pruning (disabled in endgame)
    if depth >= 3 and not board.is_check() and not is_endgame(board):
        board.push(chess.Move.null())
        try:
            null_score, _ = alpha_beta(board, depth - 3, -beta, -beta + 1, ply + 1)
            null_score = -null_score
        finally:
            board.pop()
        if null_score >= beta:
            return beta, None

    best_move = None
    best_score = -float('inf')
    original_alpha = alpha

    for move in order_moves(board, board.legal_moves, tt_move=tt_move, ply=ply):
        # Incremental accumulator update BEFORE board.push
        if USE_LEARNED_EVAL:
            accumulator.push(board, move)
        board.push(move)
        try:
            score, _ = alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
        finally:
            board.pop()
            if USE_LEARNED_EVAL:
                accumulator.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            if not board.is_capture(move) and ply < 64 and move != killer_moves[ply][0]:
                killer_moves[ply][1] = killer_moves[ply][0]
                killer_moves[ply][0] = move
                history_table[move.from_square][move.to_square] += depth * depth
            break

    # If no moves were searched, it's checkmate or stalemate
    if best_move is None:
        if board.is_check():
            return -30000 + ply, None  # checkmate
        else:
            return 0, None  # stalemate

    tt_score = best_score
    if tt_score > 20000:
        tt_score += ply
    elif tt_score < -20000:
        tt_score -= ply

    if best_score <= original_alpha:
        flag = UPPERBOUND
    elif best_score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT
    transposition_table[key] = (depth, tt_score, best_move, flag)

    return best_score, best_move


def get_move(fen: str, time_left_ms: int) -> str:
    global killer_moves, history_table, node_count, search_start_time, search_time_limit
    if len(transposition_table) > 500000:
        transposition_table.clear()
    killer_moves = [[None, None] for _ in range(64)]
    history_table = [[0] * 64 for _ in range(64)]
    node_count = 0

    board = chess.Board(fen)

    # Initialize accumulator from scratch for this position
    if USE_LEARNED_EVAL:
        accumulator.init_from_board(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    best_move = legal_moves[0]
    search_start_time = time.time()

    time_left = time_left_ms / 1000.0
    safety = 0.2
    # Conservative: assume 40 moves remaining, use time_left/40 + fraction of increment
    target = min(3.0, (time_left / 40.0) + 0.3)
    # Hard ceiling: never allocate more than what's actually left on the clock
    search_time_limit = max(0.05, min(target, time_left - safety))
    # Soft limit: don't start a new depth after using 50% of hard limit
    soft_limit = search_time_limit * 0.5

    depth = 1
    while depth <= 20:
        try:
            _, move = alpha_beta(
                board=board,
                depth=depth,
                alpha=-float('inf'),
                beta=float('inf'),
            )
            if move:
                best_move = move
            depth += 1
            # Don't start next depth if we've used most of our time
            elapsed = time.time() - search_start_time
            if elapsed > soft_limit:
                break
        except TimeoutError:
            break

    return best_move.uci()