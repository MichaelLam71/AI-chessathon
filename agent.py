from collections.abc import Hashable, Iterable

import os
import time
import chess

from nnue_eval import accumulator, evaluate_nnue, NNUEAccumulator

USE_LEARNED_EVAL = True

# Read once at import. No counters or diagnostic output when disabled.
SEARCH_STATS_ENABLED = os.environ.get("CHESS_SEARCH_STATS") == "1"
if SEARCH_STATS_ENABLED:
    import json
    import sys

search_stats: dict[str, int] = {}


def reset_search_stats() -> None:
    """Start a per-move sample; TT contents still survive between moves."""
    search_stats.clear()
    for name in (
        "q_nodes", "q_entries", "completed_depth", "completed_nodes",
        "tt_probes", "tt_hits", "tt_depth_hits", "tt_cutoffs",
        "tt_exact_returns", "tt_lower_cutoffs", "tt_upper_cutoffs",
        "beta_cutoffs", "beta_move_1", "beta_move_2", "beta_move_3", "beta_move_4plus",
        "q_beta_cutoffs", "q_beta_move_1", "q_beta_move_2",
        "q_beta_move_3", "q_beta_move_4plus", "q_stand_pat_cutoffs",
        "null_attempts", "null_cutoffs", "timed_out",
    ):
        search_stats[name] = 0


def emit_search_stats(fen: str, move: str, time_left_ms: int, elapsed_ms: float) -> None:
    """Emit one record; elapsed excludes serialization and output overhead."""
    record: dict[str, object] = dict(search_stats)
    record.update(
        fen=fen,
        move=move,
        time_left_ms=time_left_ms,
        elapsed_ms=round(elapsed_ms, 3),
        raw_node_calls=node_count,
        # Count an alpha-beta -> quiescence transition only once.
        total_nodes=node_count - search_stats["q_entries"],
        main_nodes=node_count - search_stats["q_nodes"] - search_stats["q_entries"],
    )
    print("SEARCH_STATS " + json.dumps(record, separators=(",", ":")), file=sys.stderr)

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

MATE = 30000
INF = 32000
MAX_PLY = 64
TT_SIZE = 1 << 18
HISTORY_LIMIT = 16384
# Slot -> (full key, depth, score, move, bound, generation).
transposition_table: dict[int, tuple[Hashable, int, float, chess.Move | None, int, int]] = {}
tt_generation = 0
killer_moves: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
history_table = [[[0] * 64 for _ in range(64)] for _ in range(2)]
counter_moves: dict[tuple[bool, int, int], chess.Move] = {}
node_count = 0
search_start_time = 0.0
search_time_limit = 0.0


def check_time():
    global node_count
    node_count += 1
    if (node_count & 511) == 0:  # was 2047 — check roughly every 512 nodes instead
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


def captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    return PIECE_VALUES[victim] if victim is not None else 0


def exchange_value(board: chess.Board, move: chess.Move) -> int:
    """Original, conservative legal LVA exchange estimate; 0 means keep when uncertain.

    Only simulate exchanges on the destination. Do not call NNUE while this helper
    temporarily changes the board. Checks, promotions and ambiguous LVA choices
    are deliberately treated as unknown rather than evidence for pruning.
    """
    if move.promotion or board.is_en_passant(move):
        return 0
    gains = [captured_value(board, move)]
    pushed = 0
    try:
        board.push(move)
        pushed += 1
        for _ in range(12):
            if time.time() - search_start_time > search_time_limit:
                raise TimeoutError()
            if board.is_check():
                return 0
            replies = list(board.generate_legal_captures(to_mask=chess.BB_SQUARES[move.to_square]))
            if not replies:
                break
            if any(reply.promotion for reply in replies):
                return 0
            replies.sort(key=lambda reply: PIECE_VALUES[
                board.piece_type_at(reply.from_square) or chess.PAWN
            ])
            reply = replies[0]
            if len(replies) > 1 and (
                board.piece_type_at(reply.from_square)
                == board.piece_type_at(replies[1].from_square)
            ):
                return 0
            victim = board.piece_type_at(move.to_square)
            assert victim is not None
            gains.append(PIECE_VALUES[victim] - gains[-1])
            board.push(reply)
            pushed += 1
        else:
            return 0
        for index in range(len(gains) - 1, 0, -1):
            gains[index - 1] = -max(-gains[index - 1], gains[index])
        return gains[0]
    finally:
        for _ in range(pushed):
            board.pop()


def search_draw(board: chess.Board) -> bool:
    # Avoid expensive claim/repetition work when the halfmove count rules it out.
    return (
        (board.halfmove_clock >= 4 and board.is_repetition(2))
        or (board.halfmove_clock >= 99 and board.can_claim_fifty_moves())
        or board.is_insufficient_material()
    )


def quiescence(
    board: chess.Board, alpha: float, beta: float, ply: int = 0,
    allow_draw: bool = True,
) -> float:
    if SEARCH_STATS_ENABLED:
        search_stats["q_nodes"] += 1
    check_time()
    in_check = board.is_check()
    # Resolve terminal positions before stand-pat or pruning.
    if not any(board.legal_moves):
        return -MATE + ply if in_check else 0
    if allow_draw and search_draw(board):
        return 0
    if ply >= MAX_PLY:
        return evaluate(board)

    if not in_check:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            if SEARCH_STATS_ENABLED:
                search_stats["q_stand_pat_cutoffs"] += 1
            return stand_pat
        alpha = max(alpha, stand_pat)
        moves = list(board.generate_legal_captures())
        # Include quiet promotions, which a captures-only generator omits.
        promotion_rank = chess.BB_RANK_8 if board.turn else chess.BB_RANK_1
        for move in board.generate_legal_moves(
            from_mask=board.pawns & board.occupied_co[board.turn],
            to_mask=promotion_rank & ~board.occupied,
        ):
            if move.promotion:
                moves.append(move)
    else:
        stand_pat = -INF
        moves = list(board.legal_moves)

    last_target = board.peek().to_square if board.move_stack and board.peek() else -1
    delta_allowed = not in_check and abs(alpha) < 20000 and not is_endgame(board)
    for move_index, move in enumerate(order_moves(board, moves, ply=ply), 1):
        victim = captured_value(board, move)
        attacker = board.piece_type_at(move.from_square)
        protected = (
            in_check or bool(move.promotion) or board.is_en_passant(move)
            or move.to_square == last_target
            or (board.piece_type_at(move.to_square) == chess.PAWN
                and chess.square_rank(move.to_square) in (1, 6))
        )
        if not protected:
            delta_candidate = delta_allowed and stand_pat + victim + 250 < alpha
            see_candidate = (
                attacker is not None and attacker != chess.KING
                and PIECE_VALUES[attacker] > victim
            )
            if (delta_candidate or see_candidate) and not board.gives_check(move):
                if delta_candidate:
                    continue
                if exchange_value(board, move) < -100:
                    continue
        if USE_LEARNED_EVAL:
            accumulator.push(board, move)
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1, allow_draw)
        finally:
            board.pop()
            if USE_LEARNED_EVAL:
                accumulator.pop()
        if score >= beta:
            if SEARCH_STATS_ENABLED:
                search_stats["q_beta_cutoffs"] += 1
                bucket = str(move_index) if move_index < 4 else "4plus"
                search_stats["q_beta_move_" + bucket] += 1
            return score
        alpha = max(alpha, score)
    return alpha


def order_moves(
    board: chess.Board, moves: Iterable[chess.Move] | None = None,
    tt_move: chess.Move | None = None, ply: int = 0,
) -> list[chess.Move]:
    if moves is None:
        moves = board.legal_moves
    history = history_table[board.turn]
    previous = board.peek() if board.move_stack else None
    counter = counter_moves.get((board.turn, previous.from_square, previous.to_square)) if previous else None

    def score_move(move: chess.Move) -> int:
        if move == tt_move:
            return 1000000
        if move.promotion:
            return 800000 + PIECE_VALUES[move.promotion] + captured_value(board, move)
        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square)
            # Victim first, attacker second; en passant has a pawn victim.
            return 500000 + 16 * captured_value(board, move) - PIECE_VALUES[attacker or chess.PAWN]
        if ply < MAX_PLY:
            if move == killer_moves[ply][0]:
                return 400000
            if move == killer_moves[ply][1]:
                return 390000
        if move == counter:
            return 380000
        return history[move.from_square][move.to_square]

    return sorted(moves, key=score_move, reverse=True)


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


def update_history(color: bool, move: chess.Move, bonus: int) -> None:
    row = history_table[color][move.from_square]
    old = row[move.to_square]
    row[move.to_square] = old + bonus - old * abs(bonus) // HISTORY_LIMIT


def store_tt(
    key: Hashable, depth: int, score: float, move: chess.Move | None,
    flag: int, ply: int,
) -> None:
    slot = hash(key) & (TT_SIZE - 1)
    previous = transposition_table.get(slot)
    if previous is not None:
        same_key = previous[0] == key
        fresh = previous[5] == tt_generation
        # Retain deeper results; an equal-depth exact result is more useful.
        if fresh and (
            previous[1] > depth or
            (previous[1] == depth and previous[4] == EXACT and flag != EXACT)
        ):
            return
        if same_key and move is None:
            move = previous[3]
    if score > MATE - MAX_PLY:
        score += ply
    elif score < -MATE + MAX_PLY:
        score -= ply
    transposition_table[slot] = (key, depth, score, move, flag, tt_generation)


def alpha_beta(
    board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0,
    allow_null: bool = True,
) -> tuple[float, chess.Move | None]:
    check_time()
    if ply > 0 and allow_null and search_draw(board):
        return 0, None
    if depth <= 0 or ply >= MAX_PLY:
        if SEARCH_STATS_ENABLED:
            search_stats["q_entries"] += 1
        return quiescence(board, alpha, beta, ply, allow_null), None

    # Mate-distance bounds cannot exclude a better legal mate score.
    if ply > 0:
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha, None
    original_alpha = alpha
    pv_node = beta - alpha > 1
    # Keep different rule-50 clocks and synthetic null subtrees separate.
    key = (board._transposition_key(), board.halfmove_clock, allow_null)
    entry = transposition_table.get(hash(key) & (TT_SIZE - 1))
    if SEARCH_STATS_ENABLED:
        search_stats["tt_probes"] += 1
    tt_move = None
    if entry is not None and entry[0] == key:
        if SEARCH_STATS_ENABLED:
            search_stats["tt_hits"] += 1
        _, stored_depth, score, tt_move, flag, _ = entry
        if score > MATE - MAX_PLY:
            score -= ply
        elif score < -MATE + MAX_PLY:
            score += ply
        if stored_depth >= depth:
            if SEARCH_STATS_ENABLED:
                search_stats["tt_depth_hits"] += 1
            # Root must actually search a legal move; TT still orders it first.
            if ply > 0 and (
                flag == EXACT or (flag == LOWERBOUND and score >= beta)
                or (flag == UPPERBOUND and score <= alpha)
            ):
                if SEARCH_STATS_ENABLED:
                    search_stats["tt_cutoffs"] += 1
                    name = ("tt_exact_returns", "tt_lower_cutoffs", "tt_upper_cutoffs")[flag]
                    search_stats[name] += 1
                return score, tt_move

    in_check = board.is_check()
    moves = order_moves(board, tt_move=tt_move, ply=ply)
    if not moves:
        return (-MATE + ply if in_check else 0), None

    # Null search only at non-PV nodes with material and a plausible fail-high.
    if (allow_null and ply > 0 and not pv_node and depth >= 3 and not in_check
        and abs(beta) < 20000 and board.halfmove_clock < 90
        and not is_endgame(board)
        and board.occupied_co[board.turn] & ~(board.pawns | board.kings)):
        if evaluate(board) >= beta:
            if SEARCH_STATS_ENABLED:
                search_stats["null_attempts"] += 1
            board.push(chess.Move.null())
            try:
                reduction = 2 + (depth >= 6)
                score, _ = alpha_beta(
                    board, max(0, depth - 1 - reduction), -beta, -beta + 1,
                    ply + 1, False,
                )
                score = -score
            finally:
                board.pop()
            if score >= beta and abs(score) < MATE - MAX_PLY:
                if SEARCH_STATS_ENABLED:
                    search_stats["null_cutoffs"] += 1
                store_tt(key, depth, score, tt_move, LOWERBOUND, ply)
                return score, None

    best_move = None
    best_score = -float('inf')
    quiets_searched: list[chess.Move] = []
    color = board.turn
    for move_index, move in enumerate(moves, 1):
        quiet = not board.is_capture(move) and not move.promotion
        reduce_candidate = (
            not pv_node and ply > 0 and depth >= 4 and move_index >= 5
            and quiet and not in_check and move != tt_move
            and move not in killer_moves[ply]
            and history_table[color][move.from_square][move.to_square] <= 0
            and board.piece_type_at(move.from_square) != chess.PAWN
            and not board.is_castling(move)
        )
        if USE_LEARNED_EVAL:
            accumulator.push(board, move)
        board.push(move)
        try:
            reduced = reduce_candidate and not board.is_check()
            child_depth = depth - 1
            if move_index == 1:
                score, _ = alpha_beta(board, child_depth, -beta, -alpha, ply + 1, allow_null)
                score = -score
            else:
                # Probe late moves at a zero window; every reduced alpha-raiser
                # gets its full depth back before it may cause a cutoff.
                score, _ = alpha_beta(
                    board, child_depth - int(reduced), -alpha - 1, -alpha,
                    ply + 1, allow_null,
                )
                score = -score
                if reduced and score > alpha:
                    score, _ = alpha_beta(
                        board, child_depth, -alpha - 1, -alpha, ply + 1, allow_null,
                    )
                    score = -score
                if alpha < score < beta:
                    score, _ = alpha_beta(
                        board, child_depth, -beta, -alpha, ply + 1, allow_null,
                    )
                    score = -score
        finally:
            board.pop()
            if USE_LEARNED_EVAL:
                accumulator.pop()
        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, best_score)
        if alpha >= beta:
            if SEARCH_STATS_ENABLED:
                search_stats["beta_cutoffs"] += 1
                bucket = str(move_index) if move_index < 4 else "4plus"
                search_stats["beta_move_" + bucket] += 1
            if quiet:
                if move != killer_moves[ply][0]:
                    killer_moves[ply][1] = killer_moves[ply][0]
                    killer_moves[ply][0] = move
                bonus = min(1024, depth * depth * 16)
                update_history(color, move, bonus)
                for earlier in quiets_searched:
                    update_history(color, earlier, -bonus)
                if board.move_stack and board.peek():
                    previous = board.peek()
                    counter_moves[(color, previous.from_square, previous.to_square)] = move
            break
        if quiet:
            quiets_searched.append(move)

    flag = UPPERBOUND if best_score <= original_alpha else LOWERBOUND if best_score >= beta else EXACT
    store_tt(key, depth, best_score, best_move, flag, ply)
    return best_score, best_move


def get_move(fen: str, time_left_ms: int) -> str:
    global killer_moves, node_count, search_start_time, search_time_limit, tt_generation
    tt_generation += 1
    killer_moves = [[None, None] for _ in range(MAX_PLY)]
    # Keep quiet-move experience across turns, with bounded scores and decay.
    for color_history in history_table:
        for row in color_history:
            for square in range(64):
                row[square] //= 2
    node_count = 0
    if SEARCH_STATS_ENABLED:
        reset_search_stats()

    board = chess.Board(fen)

    # Initialize accumulator from scratch for this position
    if USE_LEARNED_EVAL:
        accumulator.init_from_board(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if SEARCH_STATS_ENABLED:
            emit_search_stats(fen, "", time_left_ms, 0.0)
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
    previous_score = 0.0
    while depth <= 20:
        try:
            window = 75
            use_window = depth >= 4 and abs(previous_score) < 20000
            low = previous_score - window if use_window else -INF
            high = previous_score + window if use_window else INF
            retries = 0
            while True:
                if time.time() - search_start_time > search_time_limit:
                    raise TimeoutError()
                score, move = alpha_beta(board, depth, low, high)
                if low < score < high:
                    break
                # Widen around the previous completed score; bound retries.
                retries += 1
                window *= 2
                if retries >= 3:
                    low, high = -INF, INF
                else:
                    low, high = previous_score - window, previous_score + window
            previous_score = score
            if move:
                best_move = move
            if SEARCH_STATS_ENABLED:
                search_stats["completed_depth"] = depth
                search_stats["completed_nodes"] = node_count - search_stats["q_entries"]
            depth += 1
            # Don't start next depth if we've used most of our time
            elapsed = time.time() - search_start_time
            if elapsed > soft_limit:
                break
        except TimeoutError:
            if SEARCH_STATS_ENABLED:
                search_stats["timed_out"] = 1
            break

    if SEARCH_STATS_ENABLED:
        emit_search_stats(
            fen, best_move.uci(), time_left_ms, (time.time() - search_start_time) * 1000.0
        )
    return best_move.uci()