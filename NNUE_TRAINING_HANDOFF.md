# NNUE training handoff

## Objective and fixed configuration

Train a larger version of the best current NNUE direction using better public data only. Do not
change search, inference, features, target transformation, loss, or architecture during this run.

- Features: simple 768-dimensional piece-square features, from both player perspectives.
- Architecture: 768 -> 256 accumulator per perspective, concatenated 512 -> 32 -> 1, with
  clipped ReLU activations.
- Label: Stockfish centipawns from the side-to-move perspective.
- Target: `clip(cp, -1500, 1500) / 173.7178`.
- Objective: Huber loss with a 100 cp transition (`delta = 100 / 173.7178`).
- Seed: 19.
- Epochs: 5.

CP targets are used because the original WDL/sigmoid target compressed large advantages and did
not match the score scale consumed by search. King-bucket features were separately tested and
were worse overall, so this run must use `--features simple`.

The 5M public-data simple-CP model clearly beat the earlier simple-CP model on the fixed
400-position suite:

| Metric | Earlier CP | Public 5M |
|---|---:|---:|
| Mean cp loss | 120.36 | 88.85 |
| P90 cp loss | 421 | 191 |
| >=100 cp mistakes | 90 | 73 |
| >=200 cp blunders | 58 | 37 |
| Top-3 agreement | 56.5% | 59.75% |

Endgames regressed, so the new selector targets 30% endgames (within the requested 25-35%
range), 50% middlegames, and 20% openings. The realized proportions can differ because sampling
is bounded by what is available. The source has no per-row search-depth field, so depth cannot be
filtered directly. Restricting to shards from 2022 onward is the available reproducible proxy for
favoring a newer evaluation era; it is not a guarantee of a particular Stockfish depth/version.

## Source and selection

`prepare_public_nnue.py` streams remote Parquet from the public
[Lichess/fishnet-evals dataset](https://huggingface.co/datasets/Lichess/fishnet-evals), pinned to
revision `1b6d7c91ddef44ec89efe64e47a2e313b7648ece`. It does not download the full source.

The source `cp` field is White-relative; preparation converts it to side-to-move perspective.
Rows with mate labels are excluded. The selector balances signed CP bands, opening/middlegame/
endgame phase, and quiet/tactical/imbalanced context, suppresses close sequential duplicates, and
excludes exact simple-feature matches with the 400-position benchmark. The selected 50M rows are
materialized locally because training performs five passes over them.

## Fresh MacBook setup

The project requires Python 3.12 or newer and uses `uv`. Install Xcode command-line tools if Git
is not already available, then install `uv`, clone, and sync the locked environment:

```sh
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/advitrocks9/aichessathon-starter.git
cd aichessathon-starter
uv sync --frozen
```

If the teammate is cloning this team's fork instead, replace the clone URL with that fork's URL.
On Apple Silicon, the trainer automatically selects PyTorch MPS when available and otherwise uses
CPU. Verify the checked-out pipeline before a long run:

```sh
uv run python -m unittest tests.test_public_nnue_data tests.test_nnue_features
uv run ruff check prepare_public_nnue.py train_nnue.py convert_weights.py tools/compare_nnue_models.py
uv --system-certs run --with duckdb python prepare_public_nnue.py --inspect
```

DuckDB is deliberately injected only into the preparation command with `--with duckdb`; it is not
a competition-runtime dependency. PyTorch, NumPy, and python-chess are locked by the project.

### Hugging Face token

No token is required: `Lichess/fishnet-evals` is public. Do not put a token in this repository,
source code, a shell script, or a copied command.

If anonymous access is rate-limited and a token is desired, log in interactively so it is stored
outside the repository, then add `--hf-auth` to the inspect/preparation commands:

```sh
uvx --from huggingface-hub hf auth login
uv --system-certs run --with duckdb python prepare_public_nnue.py --inspect --hf-auth
```

DuckDB then uses Hugging Face's credential-chain provider. See Hugging Face's
[DuckDB authentication documentation](https://huggingface.co/docs/hub/en/datasets-duckdb-auth).
A read-only token is sufficient. Never share one teammate's token with another teammate.

## Final-style 50M run

Run every command from the repository root. Each output name is new and all writers refuse to
overwrite completed datasets, models, runtime weights, or benchmark reports.

### 1. Prepare the selected dataset

```sh
uv --system-certs run --with duckdb python prepare_public_nnue.py --positions 50000000 --output public_dataset_chunks_50m_final --seed 19 --min-month 2022-01 --max-scanned 600000000
```

Add `--hf-auth` only if the optional login above was performed. Output:

- `public_dataset_chunks_50m_final/chunk_public_*.npz`
- `public_dataset_chunks_50m_final/manifest.json`

The manifest records the pinned source revision, configuration, realized distributions, source
files, exclusions, and hashes. Inspect `selected_phase`, `selected_context`, and
`selected_score_bin` before training.

### 2. Train the unchanged simple-CP model

```sh
uv run python train_nnue.py --target cp --features simple --chunk-dir public_dataset_chunks_50m_final --seed 19 --epochs 5 --batch-size 256 --output weights/nnue_model_cp_simple_public_50m.pt
```

Outputs:

- Final PyTorch state dict: `weights/nnue_model_cp_simple_public_50m.pt`
- Rolling epoch checkpoint: `weights/nnue_model_cp_simple_public_50m.checkpoint.pt`

Training reads and shuffles one 50k-position chunk at a time, so it does not retain all 50M FENs
in RAM. Chunk and row orders are deterministic for each seed/epoch.

### 3. Convert to runtime NumPy weights

```sh
uv run python convert_weights.py --model weights/nnue_model_cp_simple_public_50m.pt --output weights/nnue_weights_cp_simple_public_50m.npz
```

This creates `weights/nnue_weights_cp_simple_public_50m.npz`. It does not activate the model or
change `nnue_eval.py`.

### 4. Benchmark the candidate

The existing labels already contain Stockfish scores for every legal move, so this test does not
run or require a local Stockfish binary. First rerun the active 5M model with the same checked-out
code so the comparator's source/hash guard and timing comparison are valid:

```sh
uv run python tools/position_benchmark.py test benchmarks/positions-large-labelled.json --evaluator nnue --nnue-weights weights/nnue_weights_cp_simple_public.npz --output benchmarks/results/nnue-cp-simple-public-5m-handoff-baseline-large.json
```

Then benchmark the 50M candidate:

```sh
uv run python tools/position_benchmark.py test benchmarks/positions-large-labelled.json --evaluator nnue --nnue-weights weights/nnue_weights_cp_simple_public_50m.npz --output benchmarks/results/nnue-cp-simple-public-50m-large.json
```

Compare it with that freshly measured 5M baseline:

```sh
uv run python tools/compare_nnue_models.py benchmarks/results/nnue-cp-simple-public-5m-handoff-baseline-large.json benchmarks/results/nnue-cp-simple-public-50m-large.json --output benchmarks/results/nnue-public-5m-vs-50m-comparison.md
```

The comparator also creates `benchmarks/results/nnue-public-5m-vs-50m-comparison.json`. It reports
best-move/top-3 agreement, mean/median/P90/P95 cp loss, mistake/blunder counts, mate/draw-changing
events, category results, depth/runtime, and paired bootstrap uncertainty. In particular, check
whether the endgame category recovers without degrading the overall tail-loss improvements.

Stockfish is only needed to create new labels, not to use this committed suite. If relabeling is
ever intentionally requested on macOS, install it separately (`brew install stockfish`) and pass
its path through `--stockfish`; never add the executable to competition runtime files.

## Safe resume

Preparation commits only complete chunks atomically. If interrupted, rerun the exact preparation
command with `--resume` appended. It restores the selector RNG, counters, source file and row
offset from the last complete chunk. At most the uncommitted portion of one 50k chunk is repeated.
Do not change the seed, size, benchmark, month, chunk size, or sampling settings while resuming.
Do not remove `.partial` files; they are ignored and preserve evidence of an interrupted write.

```sh
uv --system-certs run --with duckdb python prepare_public_nnue.py --positions 50000000 --output public_dataset_chunks_50m_final --seed 19 --min-month 2022-01 --max-scanned 600000000 --resume
```

Training atomically updates its dedicated checkpoint after each completed epoch. If interrupted,
rerun the exact training command with `--resume`; the incomplete epoch is restarted from the last
completed epoch. The final model path is still protected from overwrite.

```sh
uv run python train_nnue.py --target cp --features simple --chunk-dir public_dataset_chunks_50m_final --seed 19 --epochs 5 --batch-size 256 --output weights/nnue_model_cp_simple_public_50m.pt --resume
```

If a command reports mismatched resume settings or an existing final output, do not delete or
rename anything. Choose a new output dataset/model/result name or ask the repository owner.

## Storage, memory, and time

The existing 5M prepared subset occupies 154,478,496 bytes, so a comparable 50M subset is
estimated at roughly 1.5-1.7 GB. Keep at least 10 GB free for the environment, output, temporary
files, checkpoints, and operating-system headroom. Remote scanning transfers substantially more
source data than the selected output size but does not retain the full public dataset.

The chunked trainer keeps one compressed 50k-position chunk plus batches/model state in memory.
It is designed for a 16 GB MacBook; 8 GB may work but offers little unified-memory headroom for
MPS. The float model and converted runtime weights are each about 0.85 MB; the Adam checkpoint is
several MB. Five epochs over 50M rows means 250M training samples, so preparation and training can
each take many hours and possibly days depending on network speed and Mac model. Keep the laptop
on power and prevent sleep during the run.

## Protected existing artifacts

Do not overwrite or delete any of these:

- `public_dataset_chunks_5m/`
- `weights/nnue_model_cp_simple_public.pt`
- `weights/nnue_weights_cp_simple_public.npz` (the active 5M public model)
- `weights/nnue_model_cp.pt`
- `weights/nnue_weights_cp.npz`
- `weights/nnue_model_cp_king_bucket.pt`
- `weights/nnue_weights_cp_king_bucket.npz`
- Existing files under `benchmarks/results/`

Do not change `agent.py`, search code, `nnue_eval.py`, or `harness/` for this experiment. Benchmark
the 50M weights through `--nnue-weights`; promotion is a separate decision after reviewing results.

## PowerShell equivalents

The pipeline commands above are intentionally single-line and also work unchanged in PowerShell.
On Windows, install `uv` with `winget install --id=astral-sh.uv -e`, clone with Git, run
`uv sync --frozen`, and then use the same preparation/training/conversion/benchmark commands.
The Hugging Face login remains `uvx --from huggingface-hub hf auth login`.
