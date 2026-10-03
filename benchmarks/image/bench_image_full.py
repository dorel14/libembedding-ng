"""
Full image embedding benchmark with batch size sweep.
Tests all 5 models across different batch sizes to find optimal throughput.
"""
import argparse
import os
import struct
import sys
import time
import zlib

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import ImageEmbedding


def get_rss_mb():
    import subprocess
    try:
        pid = os.getpid()
        result = subprocess.run(
            ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'WorkingSetSize'],
            capture_output=True, text=True, timeout=5, check=False)
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                return int(line) / (1024 * 1024)
    except (OSError, subprocess.SubprocessError) as exc:
        # wmic is absent from recent Windows builds. Report the failure rather
        # than returning a silent 0 MB, which reads as a real measurement.
        print(f"warning: RSS probe failed ({exc}); reporting 0 MB", file=sys.stderr)
    return 0


def create_test_images(n, width=224, height=224):
    """Create n valid PNG images for benchmarking."""
    def create_minimal_png(w, h):
        def create_chunk(chunk_type, data):
            chunk = chunk_type + data
            return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
        signature = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        ihdr = create_chunk(b'IHDR', ihdr_data)
        raw_data = b''
        for y in range(h):
            raw_data += b'\x00'
            for x in range(w):
                raw_data += bytes([128, 128, 128])
        compressed = zlib.compress(raw_data)
        idat = create_chunk(b'IDAT', compressed)
        iend = create_chunk(b'IEND', b'')
        return signature + ihdr + idat + iend

    png = create_minimal_png(width, height)
    return [png for _ in range(n)]


def percentile(values, p):
    s = sorted(values)
    idx = int(p / 100.0 * len(s))
    return s[min(idx, len(s) - 1)]


def benchmark_model(model_name, images, batch_size, threads=4, warmup=2, iterations=10):
    """Benchmark a model with specific batch size."""
    rss_before = get_rss_mb()

    t0 = time.perf_counter()
    model = ImageEmbedding(
        model_name,
        threads=threads,
        batch_size=batch_size,
        offline=True,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000

    rss_after_load = get_rss_mb()

    # Warmup
    for _ in range(warmup):
        model.embed_bytes(images, batch_size=batch_size)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed_bytes(images, batch_size=batch_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    rss_after_inference = get_rss_mb()
    model.close()

    p50 = percentile(times, 50)
    p95 = percentile(times, 95)
    p99 = percentile(times, 99)

    return {
        'load_ms': load_time,
        'rss_load': rss_after_load - rss_before,
        'rss_inference': rss_after_inference - rss_before,
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'ms_image': p50 / len(images),
        'images_per_sec': 1000.0 / (p50 / len(images)) if p50 > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Full image benchmark with batch sweep')
    parser.add_argument('--images', type=int, default=16, help='Number of images')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 90)
    print("Image Embedding Benchmark — Full Batch Sweep")
    print("=" * 90)
    print(f"Images: {args.images}, Threads: {args.threads}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Create test images
    images = create_test_images(args.images)
    print(f"Created {args.images} test images")
    print()

    # Models to test
    models = [
        'openai/clip-vit-base-patch32',
        'microsoft/resnet-50',
        'open-metric-learning/unicom-vit-b-16',
        'open-metric-learning/unicom-vit-b-32',
        'nomic-ai/nomic-embed-vision-v1.5',
    ]

    # Batch sizes to test
    batch_sizes = [1, 4, 8, 16]

    all_results = []

    for model_name in models:
        print(f"{'='*90}")
        print(f"Model: {model_name}")
        print(f"{'='*90}")
        print()
        print(f"{'Batch':>8} | {'Load':>8} | {'RAM Load':>10} | {'RAM Inf':>10} | {'P50':>10} | {'P95':>10} | {'ms/img':>10} | {'img/s':>10}")
        print("-" * 95)

        for bs in batch_sizes:
            if bs > len(images):
                continue
            try:
                r = benchmark_model(model_name, images, bs, args.threads, args.warmup, args.iterations)
                r['model'] = model_name
                r['batch'] = bs
                all_results.append(r)

                print(f"{bs:>8} | {r['load_ms']:>7.0f} | {r['rss_load']:>9.0f} | {r['rss_inference']:>9.0f} | {r['p50']:>9.1f} | {r['p95']:>9.1f} | {r['ms_image']:>9.1f} | {r['images_per_sec']:>9.1f}")
            except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
                print(f"{bs:>8} | ERROR: {e}")
        print()

    # Summary
    print("=" * 90)
    print("SUMMARY — Best batch size per model")
    print("=" * 90)
    print()
    print(f"{'Model':<40} | {'Best Batch':>10} | {'P50':>10} | {'ms/img':>10} | {'img/s':>10}")
    print("-" * 90)

    for model_name in models:
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            best = min(model_results, key=lambda x: x['ms_image'])
            print(f"{model_name:<40} | {best['batch']:>10} | {best['p50']:>9.1f} | {best['ms_image']:>9.1f} | {best['images_per_sec']:>9.1f}")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Model | Batch | Load (ms) | RAM (MB) | P50 (ms) | ms/image | images/s |")
    print("|-------|-------|-----------|----------|----------|----------|----------|")
    for r in all_results:
        print(f"| {r['model']} | {r['batch']} | {r['load_ms']:.0f} | {r['rss_inference']:.0f} | {r['p50']:.1f} | {r['ms_image']:.1f} | {r['images_per_sec']:.1f} |")


if __name__ == '__main__':
    main()
