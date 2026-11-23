# Competitive benchmark (LE-11)

libembedding against SentenceTransformers and FastEmbed on the same machine, corpus and
protocol. Results and reading: [`results.md`](results.md). Raw data: `results.json`.

| File | Role |
|---|---|
| `bench_competitive.py` | the benchmark: reduced T15 mode, full LE-11 campaign, report |
| `results.md` | conditions, headline, reading, gaps above 20 % and findings, limits |
| `results_tables.md` | every per-batch, per-thread and quality table, generated from `results.json` |
| `results.json` | every cell (one subprocess each), machine and library versions, quality figures |
| `README.md` | this note |

## Protocol (full mode)

| Item | Value |
|---|---|
| Corpus | first 1000 non-empty lines of `benchmarks/datas/customers_sample.txt` (short texts) |
| Batch | 1, 8, 32, 64, 128 |
| Threads | library default, `cpu_count` (8) and `cpu_count / 2` (4) |
| Measure | one process per cell, warmup on 50 texts, 3 timed passes, median |
| Rounds | headline cell (batch 32, default threads): 3 rounds with the library order rotated; other cells: see `results.md` |
| Quality | 100 BEIR SciFact pairs (50 relevant, 50 not): cosine agreement with SentenceTransformers per text, Spearman of the pair scores against the labels |
| Models | all-MiniLM-L6-v2, bge-small-en-v1.5 (ONNX in the three libraries, GGUF Q8_0 through libembedding), SPLADE++ and bge-m3 sparse, Qwen3-Embedding-0.6B |

Thread settings are set per library (libembedding `threads=`, `torch.set_num_threads`,
FastEmbed `threads=`). Every cell records the number of OS threads of the process
(`process_threads`) and, for torch, `torch.get_num_threads()`, to check that the setting
took effect: with `threads=4` and `threads=8` the count rises by four in libembedding and
FastEmbed.

## Reproduce

Dedicated virtual environment with `sentence-transformers`, `fastembed`, `scipy`,
`psutil`; every model already in its cache (nothing is downloaded during a timed run;
`HF_HUB_OFFLINE=1` enforces it). To measure the source tree rather than the PyPI wheel:

```powershell
$env:LIBEMBEDDING_LIBRARY = "<repo>\build\Release\libembedding.dll"
$env:PYTHONPATH = "<repo>\python\src"
$env:HF_HUB_OFFLINE = "1"
python benchmarks/competitive/bench_competitive.py full --rounds 3 --grid-rounds 2
python benchmarks/competitive/bench_competitive.py report --md benchmarks/competitive/results_tables.md
```

`full` rewrites `results.json` after each cell and skips the cells already present, so an
interrupted run resumes with the same command. `--models`, `--libs`, `--batches`,
`--threads`, `--n-texts`, `--runs` restrict a run (smoke tests: `--n-texts 20 --runs 1
--rounds 1`). The reduced T15 mode is unchanged:
`python bench_competitive.py <lib> <model> <corpus>`.
