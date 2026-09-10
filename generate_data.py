import os
import random
import multiprocessing as mp
import numpy as np
import chess
import chess.engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def generate_worker(worker_id, samples_per_worker, target_depth, output_dir="dataset_chunks"):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        fens, scores, wdls = [], [], []
        chunk_idx = 0
        chunk_size = 50000
        total_generated = 0

        while total_generated < samples_per_worker:
            board = chess.Board()

            # Opening randomness: pick from top 10 engine moves for 4-9 plies
            # 4-9 range gives 50/50 white/black to move balance
            for _ in range(random.randint(4, 9)):
                if board.is_game_over() or board.can_claim_draw():
                    break
                analysis = engine.analyse(board, chess.engine.Limit(depth=4), multipv=10)
                candidate_moves = [res["pv"][0] for res in analysis if "pv" in res and res["pv"]]
                if candidate_moves:
                    board.push(random.choice(candidate_moves))
                else:
                    break

            move_count = 0
            while not board.is_game_over() and not board.can_claim_draw() and board.fullmove_number < 120:
                result = engine.play(board, chess.engine.Limit(depth=8))
                played_move = result.move

                # Check capture status BEFORE pushing move
                was_capture = board.is_capture(played_move)
                board.push(played_move)
                move_count += 1

                # Sample every 4th ply, skip early moves
                if board.fullmove_number < 5 or move_count % 4 != 0:
                    continue

                # Skip tactical noise: checks and captures produce unstable evals
                if board.is_check() or was_capture:
                    continue

                # Evaluate position at target depth
                info = engine.analyse(board, chess.engine.Limit(depth=target_depth))

                score_obj = info["score"].relative
                cp_score = score_obj.score(mate_score=10000)

                if cp_score is None or abs(cp_score) > 5000:
                    continue

                wdl = 1.0 / (1.0 + 10.0 ** (-cp_score / 400.0))

                # Original position
                fens.append(board.fen())
                scores.append(cp_score)
                wdls.append(wdl)
                total_generated += 1

                # Colour-mirrored position for zero bias
                # board.mirror() swaps pieces AND flips turn
                # Relative score stays the same (winner from STM perspective is unchanged)
                mirrored = board.mirror()
                fens.append(mirrored.fen())
                scores.append(cp_score)
                wdls.append(wdl)
                total_generated += 1

                # Progress tracking
                if total_generated % 5000 == 0:
                    print(f"Worker {worker_id}: {total_generated}/{samples_per_worker} positions generated")

                # Chunked saving
                if len(fens) >= chunk_size:
                    file_path = os.path.join(output_dir, f"chunk_w{worker_id}_{chunk_idx}.npz")
                    np.savez_compressed(
                        file_path,
                        fens=fens,
                        scores=np.array(scores, dtype=np.int16),
                        wdls=np.array(wdls, dtype=np.float32)
                    )
                    chunk_idx += 1
                    fens, scores, wdls = [], [], []

                if total_generated >= samples_per_worker:
                    break

        # Write remaining samples
        if fens:
            file_path = os.path.join(output_dir, f"chunk_w{worker_id}_{chunk_idx}.npz")
            np.savez_compressed(
                file_path,
                fens=fens,
                scores=np.array(scores, dtype=np.int16),
                wdls=np.array(wdls, dtype=np.float32)
            )

    finally:
        engine.quit()
        print(f"Worker {worker_id}: DONE. Generated {total_generated} positions in {chunk_idx + 1} chunks.")


def main():
    # 5M real positions + 5M mirrored = 10M total
    target_samples = 10 * 10**6
    target_depth = 12
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = target_samples // num_workers

    print(f"Spawning {num_workers} parallel workers for {target_samples} total samples...")
    print(f"Each worker generates {samples_per_worker} positions (including mirrors)")

    processes = []
    for w_id in range(num_workers):
        p = mp.Process(target=generate_worker, args=(w_id, samples_per_worker, target_depth))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All worker processes completed successfully.")


if __name__ == "__main__":
    main()