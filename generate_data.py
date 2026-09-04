import chess
import chess.engine
import random
import numpy as np

# Path to your local Stockfish executable
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish" 

def generate_positions(num_samples, target_depth):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    fens = []
    scores = []
    wdls = []

    games_needed = num_samples // 30  # ~30 positions per game
    
    for game_num in range(games_needed):
        if (game_num + 1) % 100 == 0:
            print(f"Game {game_num + 1}/{games_needed}, {len(fens)} positions...")
        
        board = chess.Board()

        move_count = 0
        while not board.is_game_over() and board.fullmove_number < 120:
            # Play moves using Stockfish at low depth (fast)
            result = engine.play(board, chess.engine.Limit(depth=4))
            board.push(result.move)
            move_count += 1
            
            # Skip first 4 moves (opening is too standard)
            if board.fullmove_number < 5 or move_count % 4 != 0:
                continue
            
            # Evaluate at higher depth for training labels
            info = engine.analyse(board, chess.engine.Limit(depth=target_depth))
            score_obj = info["score"].relative
            
            if score_obj.is_mate():
                cp_score = 10000 if score_obj.mate() > 0 else -10000
            else:
                cp_score = score_obj.score()
            
            # Skip positions with extreme scores (not useful for learning)
            if abs(cp_score) > 5000:
                continue
            
            wdl = 1.0 / (1.0 + 10.0 ** (-cp_score / 400.0))
            
            fens.append(board.fen())
            scores.append(cp_score)
            wdls.append(wdl)
            
            if len(fens) >= num_samples:
                break
        
        if len(fens) >= num_samples:
            break

    engine.quit()
    return fens, np.array(scores, dtype=np.int16), np.array(wdls, dtype=np.float32)

# Save generated dataset
fens, scores, wdls = generate_positions(num_samples=50000, target_depth=10)
np.savez_compressed("nnue_dataset.npz", fens=fens, scores=scores, wdls=wdls)