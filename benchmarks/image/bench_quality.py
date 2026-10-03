"""
Image embedding quality benchmark.
Tests Image→Image and Text→Image retrieval quality.

Creates synthetic images with controlled properties (colors, patterns)
to define ground truth similarity, then measures retrieval quality.
"""
import argparse
import math
import os
import struct
import sys
import zlib

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import ImageEmbedding


def create_color_image(width, height, r, g, b):
    """Create a solid color PNG image."""
    def create_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = create_chunk(b'IHDR', ihdr_data)

    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'
        for x in range(width):
            raw_data += bytes([r, g, b])

    compressed = zlib.compress(raw_data)
    idat = create_chunk(b'IDAT', compressed)
    iend = create_chunk(b'IEND', b'')
    return signature + ihdr + idat + iend


def create_pattern_image(width, height, pattern_type, color):
    """Create a PNG image with a pattern."""
    def create_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = create_chunk(b'IHDR', ihdr_data)

    r, g, b = color
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'
        for x in range(width):
            if pattern_type == 'checker':
                c = (x // 16 + y // 16) % 2
                val = (r, g, b) if c == 0 else (255 - r, 255 - g, 255 - b)
            elif pattern_type == 'gradient':
                factor = (x + y) / (width + height)
                val = (int(r * factor), int(g * factor), int(b * factor))
            elif pattern_type == 'stripes_h':
                c = (y // 16) % 2
                val = (r, g, b) if c == 0 else (128, 128, 128)
            elif pattern_type == 'stripes_v':
                c = (x // 16) % 2
                val = (r, g, b) if c == 0 else (128, 128, 128)
            else:
                val = (r, g, b)
            raw_data += bytes(val)

    compressed = zlib.compress(raw_data)
    idat = create_chunk(b'IDAT', compressed)
    iend = create_chunk(b'IEND', b'')
    return signature + ihdr + idat + iend


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_dcg(relevances, k):
    """Compute DCG@k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(similarities_with_relevance, k=10):
    """Compute NDCG@k."""
    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    relevances = [r for _, r in sorted_by_sim]
    ideal_relevances = sorted([r for _, r in similarities_with_relevance], reverse=True)
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(similarities_with_relevance, k=10, threshold=0.5):
    """Compute Recall@k."""
    total_relevant = sum(1 for _, r in similarities_with_relevance if r >= threshold)
    if total_relevant == 0:
        return 0.0
    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    relevant_in_top_k = sum(1 for _, r in sorted_by_sim[:k] if r >= threshold)
    return relevant_in_top_k / total_relevant


def compute_map(similarities_with_relevance, threshold=0.5):
    """Compute Average Precision."""
    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    num_relevant = sum(1 for _, r in sorted_by_sim if r >= threshold)
    if num_relevant == 0:
        return 0.0

    ap = 0.0
    relevant_so_far = 0
    for i, (_, rel) in enumerate(sorted_by_sim):
        if rel >= threshold:
            relevant_so_far += 1
            ap += relevant_so_far / (i + 1)
    return ap / num_relevant


def create_test_dataset():
    """Create a test dataset with known similarity relationships."""
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
    }

    patterns = ['solid', 'checker', 'gradient', 'stripes_h', 'stripes_v']

    images = []
    metadata = []

    for color_name, (r, g, b) in colors.items():
        for pattern in patterns:
            img = create_pattern_image(224, 224, pattern, (r, g, b))
            images.append(img)
            metadata.append({
                'color': color_name,
                'pattern': pattern,
                'rgb': (r, g, b),
            })

    return images, metadata


def compute_ground_truth_similarity(meta1, meta2):
    """Compute ground truth similarity based on color and pattern."""
    # Color similarity (cosine of RGB vectors)
    r1, g1, b1 = meta1['rgb']
    r2, g2, b2 = meta2['rgb']
    dot = r1 * r2 + g1 * g2 + b1 * b2
    norm1 = math.sqrt(r1**2 + g1**2 + b1**2)
    norm2 = math.sqrt(r2**2 + g2**2 + b2**2)
    color_sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    # Pattern similarity
    pattern_sim = 1.0 if meta1['pattern'] == meta2['pattern'] else 0.3

    # Combined similarity
    return color_sim * 0.7 + pattern_sim * 0.3


def main():
    parser = argparse.ArgumentParser(description='Image embedding quality benchmark')
    parser.add_argument('--k', type=int, default=10, help='Top-K for metrics')
    args = parser.parse_args()

    print("=" * 70)
    print("Image Embedding Quality Benchmark")
    print("=" * 70)
    print()

    # Create test dataset
    images, metadata = create_test_dataset()
    print(f"Test dataset: {len(images)} images")
    print(f"Colors: {len({m['color'] for m in metadata})}")
    print(f"Patterns: {len({m['pattern'] for m in metadata})}")
    print()

    models = [
        'openai/clip-vit-base-patch32',
        'microsoft/resnet-50',
    ]

    for model_name in models:
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print()

        model = ImageEmbedding(model_name, offline=True, show_download_progress=False, threads=4)

        # Embed all images
        print("Embedding images...")
        embeddings = model.embed_bytes(images, batch_size=8)
        print(f"Embedded {len(embeddings)} images, dim={len(embeddings[0])}")
        print()

        # Image→Image retrieval
        print(f"--- Image->Image Retrieval (Recall@{args.k}, NDCG@{args.k}) ---")
        recalls = []
        ndcgs = []
        maps = []

        for i, query_emb in enumerate(embeddings):
            # Compute similarities to all other images
            similarities = []
            for j, cand_emb in enumerate(embeddings):
                if i == j:
                    continue
                sim = cosine_similarity(query_emb, cand_emb)
                gt_sim = compute_ground_truth_similarity(metadata[i], metadata[j])
                similarities.append((sim, gt_sim))

            recalls.append(compute_recall(similarities, args.k))
            ndcgs.append(compute_ndcg(similarities, args.k))
            maps.append(compute_map(similarities))

        avg_recall = sum(recalls) / len(recalls)
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        avg_map = sum(maps) / len(maps)

        print(f"  Recall@{args.k}: {avg_recall:.4f}")
        print(f"  NDCG@{args.k}:   {avg_ndcg:.4f}")
        print(f"  mAP:          {avg_map:.4f}")
        print()

        model.close()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Note: Quality benchmark with synthetic images.")
    print("Real-world quality may differ significantly.")
    print("Use this for relative comparison between models, not absolute quality.")
    print()


if __name__ == '__main__':
    main()
