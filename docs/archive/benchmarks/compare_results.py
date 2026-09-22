#!/usr/bin/env python3
"""
Reads benchmark JSON results and produces a Markdown comparison table.
Usage: python3 compare_results.py <results_dir>
"""

import json
import os
import sys


def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, suffix=""):
    if val is None:
        return "n/a"
    if isinstance(val, float):
        if val >= 100:
            return f"{val:,.0f}{suffix}"
        return f"{val:.1f}{suffix}"
    return str(val)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    libs = []
    for name, filename in [
        ("libembedding (C++)", "results_libembedding.json"),
        ("fastembed-rs (Rust)", "results_fastembed_rs.json"),
        ("fastembed (Python)", "results_fastembed_py.json"),
    ]:
        r = load_result(os.path.join(results_dir, filename))
        if r:
            libs.append((name, r))

    if not libs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    headers = ["Metric"] + [name for name, _ in libs]
    rows = []

    # Model load
    rows.append(
        ["Model load (ms)"]
        + [fmt(r["benchmarks"]["model_load_ms"]["median"]) for _, r in libs]
    )

    # Single latency
    rows.append(
        ["Single text latency (ms)"]
        + [fmt(r["benchmarks"]["single_latency_ms"]["median"]) for _, r in libs]
    )

    # Batch throughput
    for bsz in ["1", "8", "32", "128", "512"]:
        vals = []
        for _, r in libs:
            bt = r["benchmarks"].get("batch_throughput", {}).get(bsz, {})
            vals.append(fmt(bt.get("median_texts_per_sec"), " texts/s"))
        rows.append([f"Batch {bsz} throughput"] + vals)

    # Memory
    rows.append(
        ["Peak RSS (MB)"]
        + [fmt(r["benchmarks"].get("memory", {}).get("peak_rss_mb")) for _, r in libs]
    )

    # Print Markdown table
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    print()
    print(fmt_row(headers))
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print(fmt_row(row))
    print()

    # Also write to file
    out_path = os.path.join(results_dir, "comparison.md")
    with open(out_path, "w") as f:
        f.write(fmt_row(headers) + "\n")
        f.write("|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n")
        for row in rows:
            f.write(fmt_row(row) + "\n")
    print(f"Table written to {out_path}")


if __name__ == "__main__":
    main()
