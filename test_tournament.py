import chess
import time

# Import your agent
from agent import get_move as my_get_move

# Import the baseline bot - uncomment the one you want to test against
#from baselines.random.baseline_random import get_move as opponent_get_move
#from baselines.greedy.baseline_greedy import get_move as opponent_get_move
#from baselines.minimax.baseline_minimax import get_move as opponent_get_move
#from baselines.numba.baseline_numba import get_move as opponent_get_move

# Simple random opponent as default fallback
import random
def random_get_move(fen, time_left_ms):
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()


def play_match(white_fn, black_fn, white_name="White", black_name="Black"):
    """Play a single game between two agents with full move logging."""
    board = chess.Board()
    move_count = 0
    white_time = 600000  # 10 minutes
    black_time = 600000

    print(f"\n  Starting position:")
    print(f"  {white_name} (White) vs {black_name} (Black)")
    print(f"  Time: {white_time/1000:.0f}s each\n")

    while not board.is_game_over() and move_count < 200:
        is_white = board.turn == chess.WHITE
        current_name = white_name if is_white else black_name
        current_time = white_time if is_white else black_time
        current_fn = white_fn if is_white else black_fn

        fen = board.fen()
        start = time.time()
        try:
            uci = current_fn(fen, current_time)
            elapsed_ms = int((time.time() - start) * 1000)
        except Exception as e:
            print(f"  ERROR: {current_name} crashed: {e}")
            return ("0-1" if is_white else "1-0"), move_count

        # Update time
        if is_white:
            white_time -= elapsed_ms
        else:
            black_time -= elapsed_ms

        # Check for illegal move
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  ILLEGAL MOVE from {current_name}: {uci}")
            return ("0-1" if is_white else "1-0"), move_count

        # Print the move
        san = board.san(move)
        move_num = (move_count // 2) + 1
        elapsed_str = f"{elapsed_ms}ms" if elapsed_ms < 1000 else f"{elapsed_ms/1000:.1f}s"
        time_remaining = white_time if is_white else black_time

        if is_white:
            print(f"  {move_num:3d}. {san:8s} ({elapsed_str:>6s}, {time_remaining/1000:.1f}s left)", end="")
        else:
            print(f"   {san:8s} ({elapsed_str:>6s}, {time_remaining/1000:.1f}s left)")

        board.push(move)
        move_count += 1

        # Check time
        if white_time <= 0:
            print(f"\n  {white_name} ran out of time!")
            return "0-1", move_count
        if black_time <= 0:
            print(f"\n  {black_name} ran out of time!")
            return "1-0", move_count

    # Print final position
    if board.turn == chess.WHITE and move_count % 2 == 1:
        print()  # newline if last move was white
    print()
    print(f"  Final position:")
    for line in str(board).split('\n'):
        print(f"    {line}")
    print()

    result = board.result()
    if board.is_checkmate():
        winner = black_name if board.turn == chess.WHITE else white_name
        print(f"  {winner} wins by checkmate!")
    elif board.is_stalemate():
        print(f"  Draw by stalemate")
    elif board.is_insufficient_material():
        print(f"  Draw by insufficient material")
    elif board.is_fifty_moves():
        print(f"  Draw by fifty-move rule")
    elif board.is_repetition():
        print(f"  Draw by repetition")
    elif move_count >= 200:
        print(f"  Draw by move limit (200)")
        result = "1/2-1/2"

    return result, move_count


def play_tournament(my_fn, opponent_fn, my_name="MyAgent", opponent_name="Opponent", num_games=6):
    """Play multiple games alternating colors."""
    my_wins = 0
    my_losses = 0
    draws = 0
    total_time_start = time.time()

    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"  GAME {game_num + 1} / {num_games}")
        print(f"{'='*60}")

        # Alternate colors
        if game_num % 2 == 0:
            result, moves = play_match(my_fn, opponent_fn, my_name, opponent_name)
            if result == "1-0":
                my_wins += 1
            elif result == "0-1":
                my_losses += 1
            else:
                draws += 1
        else:
            result, moves = play_match(opponent_fn, my_fn, opponent_name, my_name)
            if result == "0-1":
                my_wins += 1
            elif result == "1-0":
                my_losses += 1
            else:
                draws += 1

        print(f"  Result: {result} in {moves} moves")
        print(f"  Running score: {my_name} {my_wins}W-{my_losses}L-{draws}D")

    total_time = time.time() - total_time_start

    print(f"\n{'='*60}")
    print(f"  TOURNAMENT RESULTS ({num_games} games)")
    print(f"{'='*60}")
    print(f"  {my_name} vs {opponent_name}")
    print(f"  Wins:   {my_wins}")
    print(f"  Losses: {my_losses}")
    print(f"  Draws:  {draws}")
    print(f"  Score:  {my_wins + draws * 0.5} / {num_games}")
    print(f"  Win rate: {my_wins / num_games * 100:.1f}%")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Uses whatever opponent_get_move is imported at the top
    print("Testing against opponent...\n")
    play_tournament(my_get_move, opponent_get_move, "MyAgent", "Imported", num_games=6)
