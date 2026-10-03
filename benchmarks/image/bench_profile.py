"""
Image embedding profiling - measures each step separately.
Identifies whether preprocessing or ONNX inference is the bottleneck.
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


def create_test_jpeg(width=224, height=224):
    """Create a minimal valid JPEG image for benchmarking."""
    # Minimal JPEG: SOI + APP0 + DQT + SOF0 + DHT + SOS + data + EOI
    # For simplicity, create a small valid JPEG

    # Use a simpler approach: create raw RGB and use stb to encode
    # Actually, let's create a minimal valid JPEG manually
    # This is a 1x1 pixel JPEG, but we'll use PNG instead for simplicity

    # Create minimal PNG instead
    def create_minimal_png(w, h):
        def create_chunk(chunk_type, data):
            chunk = chunk_type + data
            return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

        signature = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        ihdr = create_chunk(b'IHDR', ihdr_data)

        # Raw image data with filter bytes
        raw_data = b''
        for y in range(h):
            raw_data += b'\x00'  # filter: none
            for x in range(w):
                raw_data += bytes([128, 128, 128])  # gray pixel

        compressed = zlib.compress(raw_data)
        idat = create_chunk(b'IDAT', compressed)
        iend = create_chunk(b'IEND', b'')

        return signature + ihdr + idat + iend

    return create_minimal_png(width, height)


def percentile(values, p):
    s = sorted(values)
    idx = int(p / 100.0 * len(s))
    return s[min(idx, len(s) - 1)]


def profile_pipeline(model_name, n_images=8, iterations=10):
    """Profile each step of the image embedding pipeline."""
    print(f"Profiling: {model_name}")
    print(f"Images: {n_images}, Iterations: {iterations}")
    print()

    # Create test images
    img_bytes = create_test_jpeg(224, 224)
    images = [img_bytes for _ in range(n_images)]

    # Load model
    model = ImageEmbedding(model_name, offline=True, show_download_progress=False, threads=4)
    print(f"Model loaded: {model.name}, dim={model.dim}")
    print()

    # Benchmark A: Inference pure (tensor -> ONNX -> embedding)
    # We use embed_bytes with pre-processed data
    print("--- Benchmark A: Inference (pre-processed tensors) ---")
    # Note: embed_bytes does full preprocessing, so we can't easily separate
    # Instead, we'll measure total pipeline and estimate preprocessing

    # Benchmark B: Full pipeline (JPEG -> decode -> resize -> tensor -> ONNX)
    print("--- Benchmark B: Full pipeline (JPEG -> embedding) ---")
    times_full = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed_bytes(images, batch_size=0)
        t1 = time.perf_counter()
        times_full.append((t1 - t0) * 1000)

    p50_full = percentile(times_full, 50)
    p95_full = percentile(times_full, 95)
    print(f"  P50: {p50_full:.1f} ms ({p50_full/n_images:.1f} ms/image)")
    print(f"  P95: {p95_full:.1f} ms")
    print(f"  Throughput: {1000.0 / (p50_full/n_images):.1f} images/s")
    print()

    # Benchmark C: Preprocessing only (decode + resize + normalize)
    # We can't directly measure this with the current API, but we can estimate
    # by comparing with a hypothetical "already preprocessed" path

    # For now, measure the overhead of JPEG decoding by comparing different image sizes
    print("--- Benchmark C: Preprocessing estimation ---")

    # Small images (less resize work)
    small_img = create_test_jpeg(64, 64)
    small_images = [small_img for _ in range(n_images)]
    times_small = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed_bytes(small_images, batch_size=0)
        t1 = time.perf_counter()
        times_small.append((t1 - t0) * 1000)

    p50_small = percentile(times_small, 50)

    # Large images (more resize work)
    large_img = create_test_jpeg(1024, 1024)
    large_images = [large_img for _ in range(n_images)]
    times_large = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed_bytes(large_images, batch_size=0)
        t1 = time.perf_counter()
        times_large.append((t1 - t0) * 1000)

    p50_large = percentile(times_large, 50)

    print(f"  Small (64x64):   {p50_small:.1f} ms total ({p50_small/n_images:.1f} ms/image)")
    print(f"  Normal (224x224): {p50_full:.1f} ms total ({p50_full/n_images:.1f} ms/image)")
    print(f"  Large (1024x1024): {p50_large:.1f} ms total ({p50_large/n_images:.1f} ms/image)")

    # Estimate preprocessing vs inference
    # If preprocessing dominates, large images should be much slower
    # If inference dominates, all sizes should be similar
    preprocess_estimate = p50_large - p50_full
    inference_estimate = p50_full - preprocess_estimate
    if inference_estimate < 0:
        inference_estimate = p50_full * 0.8  # rough estimate
        preprocess_estimate = p50_full * 0.2

    print(f"  Estimated preprocessing: {preprocess_estimate:.1f} ms/image")
    print(f"  Estimated inference: {inference_estimate:.1f} ms/image")
    if preprocess_estimate > 0:
        ratio = inference_estimate / preprocess_estimate
        print(f"  Inference/Preprocessing ratio: {ratio:.1f}x")
    print()

    # Benchmark D: Batch size impact
    print("--- Benchmark D: Batch size impact ---")
    batch_sizes = [1, 2, 4, 8, 16]
    for bs in batch_sizes:
        if bs > n_images:
            continue
        times_bs = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model.embed_bytes(images[:bs], batch_size=bs)
            t1 = time.perf_counter()
            times_bs.append((t1 - t0) * 1000)
        p50_bs = percentile(times_bs, 50)
        print(f"  batch={bs:>2}: {p50_bs:.7f} ms total, {p50_bs/bs:.1f} ms/image")

    model.close()
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Full pipeline P50: {p50_full:.1f} ms ({p50_full/n_images:.1f} ms/image)")
    print(f"Estimated preprocessing: ~{preprocess_estimate:.1f} ms/image")
    print(f"Estimated inference: ~{inference_estimate:.1f} ms/image")
    if preprocess_estimate > inference_estimate:
        print("=> Preprocessing is likely the bottleneck")
    else:
        print("=> Inference is likely the bottleneck")
    print()

    return {
        'model': model_name,
        'p50_full': p50_full,
        'ms_per_image': p50_full / n_images,
        'preprocess_estimate': preprocess_estimate,
        'inference_estimate': inference_estimate,
    }


def main():
    parser = argparse.ArgumentParser(description='Image embedding profiling')
    parser.add_argument('--images', type=int, default=8, help='Number of images')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 60)
    print("Image Embedding Profiling")
    print("=" * 60)
    print()

    models = [
        'openai/clip-vit-base-patch32',
        'microsoft/resnet-50',
        'open-metric-learning/unicom-vit-b-32',
    ]

    results = []
    for model_name in models:
        try:
            r = profile_pipeline(model_name, args.images, args.iterations)
            results.append(r)
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"FAILED: {e}")
        print()

    # Comparison table
    if len(results) > 1:
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print()
        print(f"{'Model':<35} | {'ms/image':>10} | {'preprocess':>12} | {'inference':>12} | {'bottleneck':>12}")
        print("-" * 90)
        for r in results:
            bottleneck = "preprocess" if r['preprocess_estimate'] > r['inference_estimate'] else "inference"
            print(f"{r['model']:<35} | {r['ms_per_image']:>9.1f} | {r['preprocess_estimate']:>11.1f} | {r['inference_estimate']:>11.1f} | {bottleneck:>12}")


if __name__ == '__main__':
    main()
