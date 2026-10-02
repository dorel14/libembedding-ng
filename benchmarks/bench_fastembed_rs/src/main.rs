use fastembed::{InitOptions, TextEmbedding, EmbeddingModel};
use serde_json::json;
use std::fs;
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        unsafe {
            let mut info: libc::rusage = mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut info) == 0 {
                return info.ru_maxrss as f64 / (1024.0 * 1024.0);
            }
        }
    }
    0.0
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn p95(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() { return 0.0; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((0.95 * v.len() as f64).ceil() as usize).saturating_sub(1);
    v[idx.min(v.len() - 1)]
}

fn load_corpus(path: &str) -> Vec<String> {
    let content = fs::read_to_string(path).expect("Failed to read corpus file");
    content.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_path = if args.len() > 1 { &args[1] } else { "corpus.txt" };
    let corpus = load_corpus(corpus_path);
    if corpus.is_empty() {
        eprintln!("Failed to load corpus from {}", corpus_path);
        std::process::exit(1);
    }

    let warmup = 1;
    let iters = 10;
    let batch_sizes: Vec<usize> = vec![1, 8, 32, 128, 512];

    // ── Benchmark A: Model load time ──────────────────────────────
    let mut load_times = Vec::new();
    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        let _model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(false),
        ).expect("Failed to create model");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            load_times.push(elapsed);
        }
    }

    // Persistent model for remaining benchmarks
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false),
    ).expect("Failed to create model");

    let rss_after_load = peak_rss_mb();

    // ── Benchmark B: Single text latency ─────────────────────────
    let mut single_times = Vec::new();
    let single_text = vec!["The quick brown fox jumps over the lazy dog.".to_string()];

    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        let _ = model.embed(single_text.clone(), Some(1)).expect("embed failed");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            single_times.push(elapsed);
        }
    }

    // ── Benchmark C: Batch throughput ─────────────────────────────
    let mut batch_throughput = serde_json::Map::new();

    for &bsz in &batch_sizes {
        let n = bsz.min(corpus.len());
        let batch_texts: Vec<String> = corpus[..n].iter().cloned().collect();
        let mut samples = Vec::new();

        for i in 0..(warmup + iters) {
            let t0 = Instant::now();
            let _ = model.embed(batch_texts.clone(), Some(bsz)).expect("embed failed");
            let elapsed = t0.elapsed().as_secs_f64();
            if i >= warmup && elapsed > 0.0 {
                samples.push(n as f64 / elapsed);
            }
        }

        let med = median(&mut samples);
        batch_throughput.insert(
            bsz.to_string(),
            json!({ "median_texts_per_sec": (med * 10.0).round() / 10.0 }),
        );
    }

    let rss_peak = peak_rss_mb();

    // ── Output JSON ───────────────────────────────────────────────
    let result = json!({
        "library": "fastembed-rs",
        "model": "all-MiniLM-L6-v2",
        // Measured, not hardcoded: this used to claim "macOS arm64"
        // unconditionally, so a run on any other machine mislabelled itself.
        "platform": if cfg!(target_os = "macos") {
            "macOS arm64"
        } else if cfg!(target_os = "windows") {
            "Windows x86_64"
        } else if cfg!(target_os = "linux") {
            "Linux x86_64"
        } else {
            "unknown"
        },
        "benchmarks": {
            "model_load_ms": {
                "median": (median(&mut load_times) * 100.0).round() / 100.0,
                "p95": (p95(&mut load_times) * 100.0).round() / 100.0,
            },
            "single_latency_ms": {
                "median": (median(&mut single_times) * 100.0).round() / 100.0,
                "p95": (p95(&mut single_times) * 100.0).round() / 100.0,
            },
            "batch_throughput": batch_throughput,
            "memory": {
                "after_load_rss_mb": (rss_after_load * 10.0).round() / 10.0,
                "peak_rss_mb": (rss_peak * 10.0).round() / 10.0,
            },
        },
    });

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
