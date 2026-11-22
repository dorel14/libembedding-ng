# Dense llama.cpp re-measurement around the v0.6.0 bump (T20) — 2026-11-22

The P5 audit found −7.4 % throughput and +10.3 % RAM for the dense GGUF backend after
the llama.cpp v0.6.0 upgrade, over 5 runs, and judged it within noise. This
re-measurement isolates the bump itself.

## Conditions

| Field | Value |
|---|---|
| Builds | `62470009~1` (before) and `62470009` (the upgrade commit, "build: upgrade llama.cpp to v0.6.0"), Release, each with its own Python binding |
| Model | `all-MiniLM-L6-v2-Q4_K_M.gguf` (second-state), dense |
| Workload | 256 short texts (`customers_sample.txt`), library default batch and threads, warmup on 32 texts, 3 timed passes (median) |
| Protocol | 10 rounds, order alternated every round, one process per measurement |
| Machine | Intel i7-1065G7, Windows 11, on mains, quiet machine |

## Results (docs/s)

| Round | before | v0.6.0 | v0.6.0 / before |
|---|---|---|---|
| 1 | 45.8 | 53.2 | disturbed |
| 2 | 146.5 | 57.8 | disturbed |
| 3 | 147.0 | 140.8 | 0.958 |
| 4 | 148.9 | 138.8 | 0.933 |
| 5 | 148.4 | 142.0 | 0.956 |
| 6 | 144.8 | 138.4 | 0.956 |
| 7 | 144.9 | 140.0 | 0.966 |
| 8 | 151.7 | 145.8 | 0.961 |
| 9 | 149.3 | 144.7 | 0.969 |
| 10 | 150.6 | 144.9 | 0.962 |

Rounds 1 and 2 were disturbed by background activity at the start of the run (both
builds collapse to about 50 docs/s) and are excluded.

## Verdict

- **Throughput: a real but small regression of about 4 %.** v0.6.0 is slower in all
  8 clean rounds (ratio 0.933 to 0.969, median 0.959).
- It stays below the 5 % threshold that would open a dedicated investigation.
- Peak RSS is unchanged (74 MB before, 75 MB after), and so is model load time
  (78 ms vs 77 ms). The +10.3 % RAM of P5 does not reproduce.
- These builds predate the runtime CPU dispatch (P2.6) and use `GGML_NATIVE`.
  The current `dev` has not been compared here.
