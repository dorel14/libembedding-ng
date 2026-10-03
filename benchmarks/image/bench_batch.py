"""
Image batch benchmark: test batch sizes 1→32 for each model.
Finds optimal batch size for throughput vs latency tradeoff.
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


def main():
    parser = argparse.ArgumentParser(description='Image batch benchmark 1→32')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 90)
    print("Image Batch Benchmark: batch size 1->32")
    print("=" * 90)
    print(f"Threads: {args.threads}, Iterations: {args.iterations}")
    print()

    # Create test images (max 32)
    max_batch = 32
    images = create_test_images(max_batch)

    models = [
        'openai/clip-vit-base-patch32',
        'microsoft/resnet-50',
        'open-metric-learning/unicom-vit-b-32',
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32]

    all_results = []

    for model_name in models:
        print(f"{'='*90}")
        print(f"Model: {model_name}")
        print(f"{'='*90}")
        print()

        model = ImageEmbedding(model_name, offline=True, show_download_progress=False, threads=args.threads)

        print(f"{'Batch':>8} | {'P50 total':>12} | {'ms/image':>10} | {'img/s':>10} | {'vs batch=1':>12}")
        print("-" * 65)

        baseline_ms_per_image = None

        for bs in batch_sizes:
            # Warmup
            for _ in range(2):
                model.embed_bytes(images[:bs], batch_size=bs)

            # Benchmark
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                model.embed_bytes(images[:bs], batch_size=bs)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            p50 = percentile(times, 50)
            ms_per_image = p50 / bs
            img_per_sec = 1000.0 / ms_per_image if ms_per_image > 0 else 0

            if baseline_ms_per_image is None:
                baseline_ms_per_image = ms_per_image
                vs_baseline = "1.00x"
            else:
                speedup = baseline_ms_per_image / ms_per_image
                vs_baseline = f"{speedup:.2f}x"

            print(f"{bs:>8} | {p50:>11.1f} | {ms_per_image:>9.1f} | {img_per_sec:>9.1f} | {vs_baseline:>12}")

            all_results.append({
                'model': model_name,
                'batch': bs,
                'p50': p50,
                'ms_per_image': ms_per_image,
                'img_per_sec': img_per_sec,
            })

        model.close()
        print()

    # Summary: optimal batch per model
    print("=" * 90)
    print("SUMMARY: Optimal batch size per model")
    print("=" * 90)
    print()
    print(f"{'Model':<40} | {'Best Batch':>10} | {'ms/image':>10} | {'img/s':>10} | {'Speedup':>10}")
    print("-" * 90)

    for model_name in models:
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            # Find best throughput (highest img/s)
            best = max(model_results, key=lambda x: x['img_per_sec'])
            baseline = next(r for r in model_results if r['batch'] == 1)
            speedup = baseline['ms_per_image'] / best['ms_per_image'] if best['ms_per_image'] > 0 else 1.0
            print(f"{model_name:<40} | {best['batch']:>10} | {best['ms_per_image']:>9.1f} | {best['img_per_sec']:>9.1f} | {speedup:>9.2f}x")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Model | Batch | P50 (ms) | ms/image | images/s | speedup |")
    print("|-------|-------|----------|----------|----------|---------|")
    for r in all_results:
        baseline = next((x for x in all_results if x['model'] == r['model'] and x['batch'] == 1), None)
        speedup = baseline['ms_per_image'] / r['ms_per_image'] if baseline and r['ms_per_image'] > 0 else 1.0
        print(f"| {r['model']} | {r['batch']} | {r['p50']:.1f} | {r['ms_per_image']:.1f} | {r['img_per_sec']:.1f} | {speedup:.2f}x |")


if __name__ == '__main__':
    main()
