"""
Vector quantization benchmark.
Measures the impact of quantizing output embeddings from float32 to int8.

This is distinct from model quantization:
- Model quantization: ONNX weights are INT8 (smaller model, faster inference)
- Vector quantization: Output embeddings are INT8 (smaller index, faster search)

Most vector DBs store float32 vectors. Quantizing to int8 can reduce index size by 4x.
"""
import argparse
import math
import os
import struct
import sys
import zlib

# Patch cffi
import cffi
import numpy as np

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import ImageEmbedding


def create_test_images(n, width=224, height=224):
    """Create n different PNG images for benchmarking."""
    def create_minimal_png(w, h, r, g, b):
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
                raw_data += bytes([r, g, b])
        compressed = zlib.compress(raw_data)
        idat = create_chunk(b'IDAT', compressed)
        iend = create_chunk(b'IEND', b'')
        return signature + ihdr + idat + iend

    # Create n images with different colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (0, 128, 255),
        (255, 0, 128), (128, 0, 255), (192, 192, 192), (64, 64, 64),
        (255, 192, 192), (192, 255, 192), (192, 192, 255), (255, 255, 192),
        (255, 192, 255), (192, 255, 255), (128, 128, 128), (255, 128, 128),
        (128, 255, 128), (128, 128, 255), (255, 255, 128),
    ]

    images = []
    for i in range(n):
        r, g, b = colors[i % len(colors)]
        images.append(create_minimal_png(width, height, r, g, b))

    return images


def quantize_to_int8(embeddings):
    """Quantize float32 embeddings to int8 using min-max scaling."""
    quantized = []
    for emb in embeddings:
        arr = np.array(emb, dtype=np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            q = np.zeros_like(arr, dtype=np.int8)
        else:
            q = ((arr - min_val) / (max_val - min_val) * 255 - 128).astype(np.int8)
        quantized.append(q)
    return quantized


def dequantize_from_int8(quantized, min_val, max_val):
    """Dequantize int8 embeddings back to float32."""
    arr = np.array(quantized, dtype=np.float32)
    return (arr + 128) / 255.0 * (max_val - min_val) + min_val


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_int8(a, b):
    """Compute cosine similarity between two int8 vectors."""
    dot = sum(int(x) * int(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(int(x) ** 2 for x in a))
    norm_b = math.sqrt(sum(int(x) ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_recall(similarities, k=10, threshold=0.5):
    """Compute Recall@k."""
    total_relevant = sum(1 for _, r in similarities if r >= threshold)
    if total_relevant == 0:
        return 0.0
    sorted_by_sim = sorted(similarities, key=lambda x: x[0], reverse=True)
    relevant_in_top_k = sum(1 for _, r in sorted_by_sim[:k] if r >= threshold)
    return relevant_in_top_k / total_relevant


def compute_ndcg(similarities, k=10):
    """Compute NDCG@k."""
    def dcg(relevances):
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    sorted_by_sim = sorted(similarities, key=lambda x: x[0], reverse=True)
    relevances = [r for _, r in sorted_by_sim]
    ideal_relevances = sorted([r for _, r in similarities], reverse=True)
    d = dcg(relevances)
    id = dcg(ideal_relevances)
    return d / id if id > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Vector quantization benchmark')
    parser.add_argument('--images', type=int, default=50, help='Number of images')
    parser.add_argument('--k', type=int, default=10, help='Top-K for metrics')
    args = parser.parse_args()

    print("=" * 70)
    print("Vector Quantization Benchmark")
    print("=" * 70)
    print(f"Images: {args.images}, K: {args.k}")
    print()

    # Create test images
    images = create_test_images(args.images)

    # Load model
    model = ImageEmbedding('openai/clip-vit-base-patch32', offline=True, show_download_progress=False, threads=4)

    # Embed all images (float32)
    print("Embedding images (float32)...")
    float32_embs = []
    for img in images:
        emb = model.embed_bytes([img], batch_size=1)
        float32_embs.append(emb[0])

    print(f"Embedded {len(float32_embs)} images, dim={len(float32_embs[0])}")
    print()

    # Quantize to int8
    print("Quantizing to int8...")
    int8_embs = quantize_to_int8(float32_embs)

    # Compute size comparison
    float32_size = sum(len(emb) * 4 for emb in float32_embs)  # 4 bytes per float32
    int8_size = sum(len(emb) for emb in int8_embs)  # 1 byte per int8

    print(f"Float32 size: {float32_size / 1024:.1f} KB")
    print(f"INT8 size: {int8_size / 1024:.1f} KB")
    print(f"Compression ratio: {float32_size / int8_size:.1f}x")
    print()

    # Measure retrieval quality
    print("--- Retrieval Quality (float32 vs int8) ---")
    float32_recalls = []
    int8_recalls = []
    float32_ndcgs = []
    int8_ndcgs = []

    for i in range(len(float32_embs)):
        # Float32 similarities
        float32_sims = []
        int8_sims = []
        for j in range(len(float32_embs)):
            if i == j:
                continue
            # Ground truth: same image = relevant (but we skip self)
            gt = 1.0 if i == j else 0.0

            f32_sim = cosine_similarity(float32_embs[i], float32_embs[j])
            i8_sim = cosine_similarity_int8(int8_embs[i], int8_embs[j])

            float32_sims.append((f32_sim, gt))
            int8_sims.append((i8_sim, gt))

        float32_recalls.append(compute_recall(float32_sims, args.k))
        int8_recalls.append(compute_recall(int8_sims, args.k))
        float32_ndcgs.append(compute_ndcg(float32_sims, args.k))
        int8_ndcgs.append(compute_ndcg(int8_sims, args.k))

    avg_f32_recall = sum(float32_recalls) / len(float32_recalls)
    avg_i8_recall = sum(int8_recalls) / len(int8_recalls)
    avg_f32_ndcg = sum(float32_ndcgs) / len(float32_ndcgs)
    avg_i8_ndcg = sum(int8_ndcgs) / len(int8_ndcgs)

    print(f"  Float32 Recall@{args.k}: {avg_f32_recall:.4f}")
    print(f"  INT8 Recall@{args.k}:    {avg_i8_recall:.4f}")
    print(f"  Float32 NDCG@{args.k}:   {avg_f32_ndcg:.4f}")
    print(f"  INT8 NDCG@{args.k}:      {avg_i8_ndcg:.4f}")
    print()

    # Similarity correlation
    print("--- Similarity Correlation ---")
    correlations = []
    for i in range(min(10, len(float32_embs))):
        f32_sims = [cosine_similarity(float32_embs[i], float32_embs[j]) for j in range(len(float32_embs)) if i != j]
        i8_sims = [cosine_similarity_int8(int8_embs[i], int8_embs[j]) for j in range(len(int8_embs)) if i != j]

        # Pearson correlation
        if len(f32_sims) > 1:
            f32_mean = sum(f32_sims) / len(f32_sims)
            i8_mean = sum(i8_sims) / len(i8_sims)
            cov = sum((f - f32_mean) * (i - i8_mean) for f, i in zip(f32_sims, i8_sims))
            f32_std = math.sqrt(sum((f - f32_mean) ** 2 for f in f32_sims))
            i8_std = math.sqrt(sum((i - i8_mean) ** 2 for i in i8_sims))
            corr = cov / (f32_std * i8_std) if f32_std > 0 and i8_std > 0 else 0
            correlations.append(corr)

    avg_corr = sum(correlations) / len(correlations) if correlations else 0
    print(f"  Pearson correlation (float32 vs int8 similarities): {avg_corr:.4f}")
    print()

    model.close()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Vector quantization: float32 -> int8")
    print(f"Size reduction: {float32_size / int8_size:.1f}x")
    print(f"Recall impact: {avg_i8_recall / avg_f32_recall:.2f}x" if avg_f32_recall > 0 else "N/A")
    print(f"NDCG impact: {avg_i8_ndcg / avg_f32_ndcg:.2f}x" if avg_f32_ndcg > 0 else "N/A")
    print(f"Similarity correlation: {avg_corr:.4f}")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Metric | float32 | INT8 | Ratio |")
    print("|--------|---------|------|-------|")
    print(f"| Size | {float32_size / 1024:.1f} KB | {int8_size / 1024:.1f} KB | {float32_size / int8_size:.1f}x |")
    print(f"| Recall@{args.k} | {avg_f32_recall:.4f} | {avg_i8_recall:.4f} | {avg_i8_recall / avg_f32_recall:.2f}x |" if avg_f32_recall > 0 else "")
    print(f"| NDCG@{args.k} | {avg_f32_ndcg:.4f} | {avg_i8_ndcg:.4f} | {avg_i8_ndcg / avg_f32_ndcg:.2f}x |" if avg_f32_ndcg > 0 else "")


if __name__ == '__main__':
    main()
