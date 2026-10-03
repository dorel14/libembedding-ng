"""
Image quantization benchmark: FP32 vs INT8.
Measures performance and quality difference.
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


def benchmark_model(model_name, images, batch_size, warmup=2, iterations=10):
    """Benchmark a model with specific batch size."""
    rss_before = get_rss_mb()

    t0 = time.perf_counter()
    model = ImageEmbedding(model_name, offline=True, show_download_progress=False, threads=4)
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

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(0.95 * len(times))]

    return {
        'load_ms': load_time,
        'rss_load': rss_after_load - rss_before,
        'rss_inference': rss_after_inference - rss_before,
        'p50': p50,
        'p95': p95,
        'ms_image': p50 / len(images),
        'images_per_sec': 1000.0 / (p50 / len(images)) if p50 > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Image quantization benchmark')
    parser.add_argument('--images', type=int, default=16, help='Number of images')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 80)
    print("Image Quantization Benchmark: FP32 vs INT8")
    print("=" * 80)
    print(f"Images: {args.images}, Batch: {args.batch_size}, Iterations: {args.iterations}")
    print()

    # Create test images
    images = create_test_images(args.images)

    models = [
        ('openai/clip-vit-base-patch32', 'FP32'),
        ('openai/clip-vit-base-patch32-quantized', 'INT8'),
    ]

    results = []
    for model_name, label in models:
        print(f"--- {label} ({model_name}) ---")
        try:
            r = benchmark_model(model_name, images, args.batch_size, iterations=args.iterations)
            r['model'] = model_name
            r['label'] = label
            results.append(r)

            print(f"  Load time: {r['load_ms']:.0f} ms")
            print(f"  RAM load: {r['rss_load']:.0f} MB")
            print(f"  RAM inference: {r['rss_inference']:.0f} MB")
            print(f"  P50: {r['p50']:.1f} ms ({r['ms_image']:.1f} ms/image)")
            print(f"  P95: {r['p95']:.1f} ms")
            print(f"  Throughput: {r['images_per_sec']:.1f} images/s")
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"  FAILED: {e}")
        print()

    # Comparison
    if len(results) == 2:
        fp32 = results[0]
        int8 = results[1]

        print("=" * 80)
        print("COMPARISON: INT8 vs FP32")
        print("=" * 80)
        print()
        print(f"{'Metric':<25} | {'FP32':>12} | {'INT8':>12} | {'Speedup':>10}")
        print("-" * 65)
        print(f"{'Load time (ms)':<25} | {fp32['load_ms']:>11.0f} | {int8['load_ms']:>11.0f} | {fp32['load_ms']/int8['load_ms']:>9.2f}x")
        print(f"{'RAM load (MB)':<25} | {fp32['rss_load']:>11.0f} | {int8['rss_load']:>11.0f} | {fp32['rss_load']/int8['rss_load']:>9.2f}x")
        print(f"{'RAM inference (MB)':<25} | {fp32['rss_inference']:>11.0f} | {int8['rss_inference']:>11.0f} | {fp32['rss_inference']/int8['rss_inference']:>9.2f}x")
        print(f"{'P50 (ms)':<25} | {fp32['p50']:>11.1f} | {int8['p50']:>11.1f} | {fp32['p50']/int8['p50']:>9.2f}x")
        print(f"{'ms/image':<25} | {fp32['ms_image']:>11.1f} | {int8['ms_image']:>11.1f} | {fp32['ms_image']/int8['ms_image']:>9.2f}x")
        print(f"{'images/s':<25} | {fp32['images_per_sec']:>11.1f} | {int8['images_per_sec']:>11.1f} | {int8['images_per_sec']/fp32['images_per_sec']:>9.2f}x")
        print()

        if int8['ms_image'] < fp32['ms_image'] * 0.9:
            print("=> INT8 is FASTER than FP32. Quantization is beneficial.")
        elif int8['ms_image'] > fp32['ms_image'] * 1.1:
            print("=> INT8 is SLOWER than FP32. Quantization is NOT beneficial.")
        else:
            print("=> INT8 and FP32 are comparable.")
        print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Model | Load (ms) | RAM (MB) | P50 (ms) | ms/image | images/s |")
    print("|-------|-----------|----------|----------|----------|----------|")
    for r in results:
        print(f"| {r['label']} | {r['load_ms']:.0f} | {r['rss_inference']:.0f} | {r['p50']:.1f} | {r['ms_image']:.1f} | {r['images_per_sec']:.1f} |")


if __name__ == '__main__':
    main()
