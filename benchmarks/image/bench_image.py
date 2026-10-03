"""
Image embedding benchmark.
Compares available image models on latency, RAM, and throughput.

Models tested:
- openai/clip-vit-base-patch32 (CLIP, 512-dim)
- microsoft/resnet-50 (ResNet, 2048-dim)
- open-metric-learning/unicom-vit-b-16 (UNICOM, 768-dim)
- open-metric-learning/unicom-vit-b-32 (UNICOM, 512-dim)
- nomic-ai/nomic-embed-vision-v1.5 (Nomic, 768-dim)
"""
import argparse
import os
import subprocess
import sys
import time

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import ImageEmbedding, list_image_models


def get_rss_mb():
    """Get current process RSS in MB."""
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


def generate_synthetic_images(n, width=224, height=224):
    """Generate synthetic JPEG-like images for benchmarking.

    Creates random byte arrays that simulate image data.
    In production, use real images.
    """
    import random
    images = []
    for _ in range(n):
        # Create a synthetic "image" - random bytes of typical JPEG size
        # Real implementation would load actual images
        size = random.randint(10000, 50000)  # 10-50KB typical JPEG
        img = bytes([random.randint(0, 255) for _ in range(size)])
        images.append(img)
    return images


def benchmark_image_model(model_name, images, threads, batch_size, warmup=2, iterations=10):
    """Benchmark an image embedding model."""
    rss_before = get_rss_mb()

    # Load model
    t0 = time.perf_counter()
    model = ImageEmbedding(
        model_name,
        threads=threads,
        batch_size=batch_size,
        offline=False,
        show_download_progress=True,
    )
    load_time = (time.perf_counter() - t0) * 1000

    rss_after_load = get_rss_mb()

    # Warmup
    for _ in range(warmup):
        try:
            model.embed_bytes(images, batch_size=batch_size)
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"  Warmup error: {e}")
            # If embed_bytes fails (invalid image data), just measure load time
            model.close()
            return {
                'load_ms': load_time,
                'rss_delta': rss_after_load - rss_before,
                'error': str(e),
            }

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed_bytes(images, batch_size=batch_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    info = model.info()
    model.close()

    # Compute stats
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(0.95 * len(times))]
    mean = sum(times) / len(times)

    return {
        'load_ms': load_time,
        'rss_delta': rss_after_load - rss_before,
        'p50': p50,
        'p95': p95,
        'mean': mean,
        'ms_image': p50 / len(images),
        'images_per_sec': 1000.0 / (p50 / len(images)) if p50 > 0 else 0,
        'dim': info.max_length,  # Using max_length as proxy
        'error': None,
    }


def main():
    parser = argparse.ArgumentParser(description='Image embedding benchmark')
    parser.add_argument('--images', type=int, default=10, help='Number of synthetic images')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0=default)')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 80)
    print("Image Embedding Benchmark")
    print("=" * 80)
    print(f"Images: {args.images}, Threads: {args.threads}, Batch: {args.batch_size or 'default'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Get available models
    models = list_image_models()
    print(f"Available models: {len(models)}")
    for m in models:
        print(f"  - {m.model_name} (dim={m.dim})")
    print()

    # Generate synthetic images
    images = generate_synthetic_images(args.images)
    print(f"Generated {args.images} synthetic images")
    print()

    results = []
    for m in models:
        print(f"--- {m.model_name} ---")
        try:
            r = benchmark_image_model(
                m.model_name, images, args.threads, args.batch_size,
                args.warmup, args.iterations
            )
            r['model'] = m.model_name
            r['dim'] = m.dim
            results.append(r)

            print(f"  Load time: {r['load_ms']:.0f} ms")
            print(f"  RAM delta: {r['rss_delta']:.0f} MB")
            if r['error']:
                print(f"  Error: {r['error']}")
            else:
                print(f"  P50: {r['p50']:.1f} ms ({r['ms_image']:.1f} ms/image)")
                print(f"  P95: {r['p95']:.1f} ms")
                print(f"  Throughput: {r['images_per_sec']:.1f} images/s")
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"  FAILED: {e}")
            results.append({
                'model': m.model_name,
                'dim': m.dim,
                'error': str(e),
            })
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<45} | {'Dim':>5} | {'Load':>8} | {'RAM':>6} | {'P50':>8} | {'ms/img':>8} | {'img/s':>8}")
    print("-" * 100)
    for r in results:
        if r.get('error'):
            print(f"{r['model']:<45} | {r.get('dim', 0):>5} | {'ERROR':>8} | {'---':>6} | {'---':>8} | {'---':>8} | {'---':>8}")
        else:
            print(f"{r['model']:<45} | {r.get('dim', 0):>5} | {r['load_ms']:>7.0f} | {r['rss_delta']:>5.0f} | {r['p50']:>7.1f} | {r['ms_image']:>7.1f} | {r['images_per_sec']:>7.1f}")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Model | Dim | Load (ms) | RAM (MB) | P50 (ms) | ms/image | images/s |")
    print("|-------|-----|-----------|----------|----------|----------|----------|")
    for r in results:
        if not r.get('error'):
            print(f"| {r['model']} | {r.get('dim', 0)} | {r['load_ms']:.0f} | {r['rss_delta']:.0f} | {r['p50']:.1f} | {r['ms_image']:.1f} | {r['images_per_sec']:.1f} |")


if __name__ == '__main__':
    main()
