import time
import chess

from features import featurize, material_score_stm
from features import PIECE_TYPES, phase_of

USE_LEARNED_EVAL = False

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
killer_moves = [[None, None] for _ in range(64)]  # 2 killer moves per depth
history_table = [[0] * 64 for _ in range(64)]  # from_sq -> to_sq bonus

LEARNED_PAWN_OPENING = [0, 0, 0, 0, 0, 0, 0, 0, -104, -62, -77, -88, -67, -41, -24, -12, -101, -51, -57, -61, -46, -33, -15, -6, -69, -64, -50, -43, -35, -46, -3, -3, -81, -66, -54, -49, -43, -56, -25, -5, -97, -72, -64, -73, -65, -43, -12, -19, -106, -79, -81, -94, -86, -47, -29, -29, 0, 0, 0, 0, 0, 0, 0, 0]
LEARNED_KNIGHT_OPENING = [-56, -187, -304, -100, -212, -201, -192, -154, -110, -94, -103, -175, -175, -12, -186, -222, -164, -155, -154, -140, -62, -160, -140, -150, -172, -136, -163, -169, -138, -145, -112, -170, -202, -157, -165, -175, -152, -133, -137, -155, -214, -171, -181, -95, -160, -163, -167, -168, -124, -184, -160, -200, -191, -30, -180, -244, 52, -213, -306, -122, -201, -215, -193, -128]
LEARNED_BISHOP_OPENING = [-169, -77, -177, -356, -311, -194, -268, -220, -119, -135, -87, -177, -164, -183, -152, -249, -102, -139, -144, -147, -146, -153, -128, -96, -190, -131, -172, -111, -54, -151, -152, -174, -168, -156, -164, -129, -117, -153, -160, -159, -131, -148, -184, -153, -141, -176, -137, -125, -110, -138, -85, -179, -165, -267, -151, -211, -173, -22, -176, -358, -259, -182, -141, -188]
LEARNED_ROOK_OPENING = [-440, -397, -411, -353, -336, -378, -366, -421, -363, -254, -372, -294, -240, -355, -299, -427, -356, -246, -375, -341, -304, -370, -257, -294, -325, -448, -332, -305, -347, -438, -239, -400, -388, -463, -274, -284, -300, -423, -305, -448, -368, -253, -394, -322, -319, -399, -255, -270, -313, -214, -442, -300, -333, -354, -344, -477, -418, -383, -380, -355, -340, -390, -364, -391]
LEARNED_QUEEN_OPENING = [-579, -507, -511, -447, -395, -271, -490, -407, -520, -448, -424, -454, -449, -392, -316, -323, -375, -431, -429, -460, -447, -450, -433, -211, -443, -454, -487, -429, -472, -506, -395, -402, -454, -463, -531, -485, -487, -491, -379, -444, -454, -462, -516, -512, -495, -455, -452, -397, -506, -527, -457, -484, -466, -404, -516, -417, -849, -573, -575, -487, -457, -457, -497, -619]
LEARNED_KING_OPENING = [4, -13, -40, -103, 2, -91, -2, 5, -241, -26, -113, -128, -44, -97, -41, -131, 98, -131, -95, 195, -71, -199, -163, 53, 191, 320, -22, 424, 193, 316, -251, -352, 136, 366, 11, 411, 171, 69, -279, -270, -216, 87, -280, 7, 43, -38, 41, 24, -240, 16, -131, -3, 16, 39, 85, -25, 63, 100, 74, -2, 82, -11, 117, 93]
LEARNED_PAWN_ENDGAME = [0, 0, 0, 0, 0, 0, 0, 0, 120, 88, 48, 72, 116, 116, 79, -53, 96, -47, 23, 5, 57, -53, -6, -60, -18, 103, -74, 21, 15, 60, 31, -84, 12, 80, -88, 7, 23, 72, 57, -71, 100, 10, 19, 4, 97, -12, -12, -51, 132, 96, 82, -32, 110, 70, 108, -26, 0, 0, 0, 0, 0, 0, 0, 0]
LEARNED_KNIGHT_ENDGAME = [-481, -200, -215, -281, -275, -325, -118, 9, -99, -344, 0, -22, -110, 98, -266, -34, -161, -349, -56, 107, -62, -4, -131, -431, -90, -26, 26, 33, 10, 111, -9, -48, -6, -100, -84, 11, 21, 104, -196, 22, -128, -292, -53, 1, -47, -82, -125, -223, -35, -489, 33, -46, -134, 66, -220, -12, -596, -183, -130, -312, -196, -328, -274, -31]
LEARNED_BISHOP_ENDGAME = [7, -271, -70, -70, 78, -68, 155, 74, -50, 9, -81, 127, -41, 13, -20, 103, 103, -65, 50, -23, 84, 28, 116, -84, 27, 88, 40, 71, -137, -24, -148, 83, 40, 93, 42, 42, -110, -32, -111, 53, 0, -91, 17, -63, -19, 40, -5, 13, -39, -96, -104, 94, -81, 15, -77, -181, -87, -187, -103, -122, 64, -138, 180, 39]
LEARNED_ROOK_ENDGAME = [1, 50, 100, -29, 50, 52, -71, -60, -78, -59, 39, 66, -83, 58, -133, 10, 34, 6, -2, 85, -42, 19, 45, 23, -49, 47, -178, -45, -34, 103, -30, -12, 54, -26, -211, -85, -70, 104, -42, -30, -143, 35, -27, 22, -102, -25, -25, -99, -129, -84, 35, -45, -58, 16, -117, 39, -18, -32, 29, -33, -40, 36, -77, -68]
LEARNED_QUEEN_ENDGAME = [-118, -54, -186, -306, -249, -506, -208, -131, -23, -238, -118, -47, -175, -149, -645, 96, -289, -383, -240, -151, 44, 72, -168, -385, -267, -137, 59, -193, 14, -23, -205, -97, -160, -206, 43, 18, 4, 63, -214, 36, -358, -350, 143, -95, 70, 85, 73, -58, -200, -74, -196, -80, -146, -103, -57, 74, 455, 107, -80, -213, -52, -249, 30, 25]
LEARNED_KING_ENDGAME = [-26, 122, 33, 14, -28, 103, -31, -84, 121, -8, -65, 47, -72, 13, 7, -36, -141, 61, -28, -152, 1, -3, -37, -81, 74, 103, -20, -19, 1, 9, 47, -10, 65, 109, 6, -4, -38, 14, 125, -25, -10, 41, -13, -39, -67, -83, -55, -33, 140, 15, -22, -13, -71, -24, -35, -73, 49, 64, 55, 39, -26, 106, -37, -74]

OPENING_TABLES = [LEARNED_PAWN_OPENING, LEARNED_KNIGHT_OPENING, LEARNED_BISHOP_OPENING,
                  LEARNED_ROOK_OPENING, LEARNED_QUEEN_OPENING, LEARNED_KING_OPENING]
ENDGAME_TABLES = [LEARNED_PAWN_ENDGAME, LEARNED_KNIGHT_ENDGAME, LEARNED_BISHOP_ENDGAME,
                  LEARNED_ROOK_ENDGAME, LEARNED_QUEEN_ENDGAME, LEARNED_KING_ENDGAME]

OPENING_WEIGHTS = sum(OPENING_TABLES, [])
ENDGAME_WEIGHTS = sum(ENDGAME_TABLES, [])


def evaluate_learned(board: chess.Board) -> int:
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    stm = board.turn
    phase = phase_of(board)
    opening_factor = 1.0 - phase
    endgame_factor = phase

    pst_score = 0.0
    for square, piece in board.piece_map().items():
        pt_idx = PIECE_TYPES.index(piece.piece_type)
        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        idx = pt_idx * 64 + sq
        sign = 1.0 if piece.color == stm else -1.0
        weight = OPENING_WEIGHTS[idx] * opening_factor + ENDGAME_WEIGHTS[idx] * endgame_factor
        pst_score += sign * weight

    base = material_score_stm(board)

    # Mobility bonus: more legal moves = better position
    mobility = len(list(board.legal_moves))
    mobility_bonus = mobility * 5

    # Passed pawn bonus: pawns with no opposing pawns in front
    passed_pawn_bonus = 0
    for sq in board.pieces(chess.PAWN, stm):
        rank = chess.square_rank(sq) if stm == chess.WHITE else 7 - chess.square_rank(sq)
        file = chess.square_file(sq)
        is_passed = True
        for opp_sq in board.pieces(chess.PAWN, not stm):
            opp_file = chess.square_file(opp_sq)
            opp_rank = chess.square_rank(opp_sq) if stm == chess.WHITE else 7 - chess.square_rank(opp_sq)
            if abs(opp_file - file) <= 1 and opp_rank > rank:
                is_passed = False
                break
        if is_passed:
            passed_pawn_bonus += rank * rank * 3  # quadratic bonus, further = much better

    return int(base + pst_score + mobility_bonus + passed_pawn_bonus)


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
    return evaluate_learned(board) if USE_LEARNED_EVAL else evaluate_classical(board)


def evaluate_classical(board: chess.Board) -> int:
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

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
            opp_rank = chess.square_rank(opp_sq)
            opp_file = chess.square_file(opp_sq)
            if abs(opp_file - file) <= 1 and opp_rank > rank:
                is_passed = False
                break
        if is_passed:
            score += rank * rank * 3

    for sq in board.pieces(chess.PAWN, chess.BLACK):
        rank = 7 - chess.square_rank(sq)
        file = chess.square_file(sq)
        is_passed = True
        for opp_sq in board.pieces(chess.PAWN, chess.WHITE):
            opp_rank = 7 - chess.square_rank(opp_sq)
            opp_file = chess.square_file(opp_sq)
            if abs(opp_file - file) <= 1 and opp_rank > rank:
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

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    for move in order_moves(board):
        if not board.is_capture(move):
            continue
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
        finally:
            board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def order_moves(board: chess.Board, tt_move=None, depth=0):
    def score_move(move: chess.Move):
        if tt_move and move == tt_move:
            return 100000
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            return 50000 + (victim_val - attacker_val)
        # Killer move bonus
        if depth < 64:
            if move == killer_moves[depth][0]:
                return 40000
            if move == killer_moves[depth][1]:
                return 39000
        # History heuristic
        return history_table[move.from_square][move.to_square]

    return sorted(board.legal_moves, key=score_move, reverse=True)


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float, ply: int = 0):
    if time.time() - start_time > time_limit:
        raise TimeoutError()

    # Check extension: search one deeper when in check
    in_check = board.is_check()
    if in_check:
        depth += 1

    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit), None

    key = board._transposition_key()
    tt_entry = transposition_table.get(key)
    tt_move = None
    if tt_entry:
        stored_depth, stored_score, tt_move = tt_entry
        if stored_depth >= depth:
            return stored_score, tt_move

    # Null move pruning (disabled in endgame and when in check)
    if depth >= 3 and not in_check and not is_endgame(board):
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
    moves_searched = 0

    for move in order_moves(board, tt_move=tt_move, depth=ply):
        board.push(move)
        try:
            # Late move reductions: search later moves at reduced depth
            if (moves_searched >= 4 and depth >= 3 and
                not in_check and not board.is_check() and
                not board.is_capture(move)):
                # Reduced depth search first
                score, _ = alpha_beta(board, depth - 2, -alpha - 1, -alpha, start_time, time_limit, ply + 1)
                score = -score
                if score > alpha:
                    # Re-search at full depth if it looks promising
                    score, _ = alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit, ply + 1)
                    score = -score
            else:
                score, _ = alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit, ply + 1)
                score = -score
        finally:
            board.pop()

        moves_searched += 1

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            # Update killer moves and history for beta cutoff on quiet moves
            if not board.is_capture(move) and ply < 64:
                killer_moves[ply][1] = killer_moves[ply][0]
                killer_moves[ply][0] = move
                history_table[move.from_square][move.to_square] += depth * depth
            break

    transposition_table[key] = (depth, best_score, best_move)
    return best_score, best_move


def get_move(fen: str, time_left_ms: int) -> str:
    global killer_moves, history_table
    # Don't clear TT - positions from previous moves are still valid
    if len(transposition_table) > 500000:
        transposition_table.clear()
    # Reset move ordering heuristics each turn
    killer_moves = [[None, None] for _ in range(64)]
    history_table = [[0] * 64 for _ in range(64)]

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    best_move = legal_moves[0]
    start_time = time.time()

    # Time management: use more time early, less when low
    allocated_seconds = max(0.1, min(8.0, (time_left_ms / 1000.0) * 0.06))
    if time_left_ms < 10000:
        allocated_seconds = 0.03
    elif time_left_ms < 20000:
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

    # Anti-repetition: if best move causes repetition and we have alternatives, pick the next best
    board.push(best_move)
    if board.is_repetition(2):
        board.pop()
        # Search for the best non-repeating move
        second_best_move = None
        second_best_score = -float('inf')
        for move in legal_moves:
            if move == best_move:
                continue
            board.push(move)
            if not board.is_repetition(2):
                score = -evaluate(board)
                if score > second_best_score:
                    second_best_score = score
                    second_best_move = move
            board.pop()
        # Only use the alternative if it's not terrible (within 200 centipawns of best)
        if second_best_move is not None:
            board.push(best_move)
            best_score = -evaluate(board)
            board.pop()
            if second_best_score > best_score - 200:
                best_move = second_best_move
    else:
        board.pop()

    return best_move.uci()