---
nav_exclude: true
---

# Benchmark methodology

Version: 1.0.0

A performance number is only worth something if it can be reproduced. This page sets
the protocol the benchmarks under `benchmarks/` follow and the metadata every report
must carry. Français : [Méthodologie benchmark](../benchmark_methodology.html).

## 1. Protocol

1. **One subprocess per backend.** Peak memory (RSS) is a per-process high-water
   mark: it can only rise. A process that already ran one backend can never report a
   lower peak for the next, so a subprocess per configuration is the only way to
   attribute memory. It also isolates a backend that crashes or runs out of memory.
2. **Warmup separate from the timed run.** Warmup runs on the same instance that is
   measured (lazy init and the thread pool are per instance), but on its own slice of
   documents, never on the timed one.
3. **Model load time is measured on its own**, before the warmup.
4. **Alternating rounds for an A/B.** To compare two configurations (or two builds),
   alternate A, B, A, B, ... instead of "all A then all B": thermal drift and
   background load then hit both the same way.
5. **Median over at least 10 rounds** for a performance conclusion, plus p95 for
   latency. A mean is pulled around by a single outlier.
6. **Noise threshold: a difference under 5 % is noise.** Claim neither a gain nor a
   regression below 5 % unless the measurement is repeated and the gap holds.
7. **Idle machine, no parallel builds.** Record any background load.
8. **Never assume a quantization is faster: measure it.**

## 2. Mandatory metadata

Every report (JSON and Markdown) carries these fields. A missing, `unknown` or zero
field is flagged by `missing_fields()`.

| Group | Fields |
|---|---|
| Source | git commit (and dirty flag), libembedding version |
| Machine | OS, CPU model, physical / logical cores, RAM, detected instruction sets |
| Software | Python version, ONNX Runtime version; llama.cpp / ggml when the bindings expose it |
| Run | threads, batch, warmup, iterations, format (onnx/gguf), quantization, dataset |
| Results | p50 and p95, load time, peak RSS |

A report whose commit is *dirty* measures uncommitted code: do not publish it as a
reference.

## 3. Using `benchmarks/_benchmeta.py`

```python
import _benchmeta as bm

machine = bm.collect_machine_metadata()
run = bm.RunParams(threads=0, batch=32, warmup=50, iterations=10,
                   format="onnx", quantization="fp32", dataset="synthetic-1000")

model, load_ms = bm.timed_load(lambda: TextEmbedding(name))
# ... warmup, then the timed loop -> samples_ms
stats = bm.summarize_latencies(samples_ms)    # n, min_ms, p50_ms, p95_ms, max_ms
peak = bm.peak_rss_mb()                       # inside the backend's subprocess

report = bm.build_report(machine, run, results={"p50_ms": stats["p50_ms"],
                                                "load_ms": load_ms, "peak_mb": peak})
assert not bm.missing_fields(report)
bm.write_json(report, "result.json")          # LF, no BOM
print(bm.to_markdown(report))                 # "Run metadata" Markdown block
```

- `threads=0` means "library default".
- `batch` takes an integer or the list of sizes of a sweep.
- `percentile()` interpolates linearly between ranks; `p50` and `p95` derive from it.
- `is_noise(baseline, candidate)` applies the 5 % threshold of section 1.
- The helpers do not raise when a source is missing (no DLL, no git): the field is
  `unknown` and shows up in `missing_fields()`.

Scripts already wired: `benchmarks/sparse/bench_sparse_backend_comparison.py`
(`--output-json` option, p50/p95 table columns) and
`benchmarks/quantization/bench_quantization.py` (`metadata` key of the JSON).

## 4. Known limits

- The bindings expose no ggml or llama.cpp version number: `lembed_llama_version()`
  only says whether llama.cpp is enabled. The vendored snapshot is described in
  `third_party/README.md`.
- A p95 over few iterations (5) interpolates between the two largest samples; it only
  becomes statistically meaningful from about twenty rounds.
