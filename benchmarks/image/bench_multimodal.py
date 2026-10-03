"""
Multimodal quality benchmark for image embedding.
Tests Image→Image retrieval quality.

NOTE: CLIP also supports Text→Image and Image→Text retrieval via a shared
text/image embedding space. However, this requires the text encoder branch
of CLIP, which is not yet exposed in libembedding's ImageEmbedding API.

Once the text encoder is exposed, this benchmark should be extended to:
- Text→Image: query text → find matching images
- Image→Text: query image → find matching captions
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


def compute_recall(similarities_with_relevance, k=10, threshold=0.5):
    """Compute Recall@k."""
    total_relevant = sum(1 for _, r in similarities_with_relevance if r >= threshold)
    if total_relevant == 0:
        return 0.0
    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    relevant_in_top_k = sum(1 for _, r in sorted_by_sim[:k] if r >= threshold)
    return relevant_in_top_k / total_relevant


def compute_ndcg(similarities_with_relevance, k=10):
    """Compute NDCG@k."""
    def dcg(relevances):
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    relevances = [r for _, r in sorted_by_sim]
    ideal_relevances = sorted([r for _, r in similarities_with_relevance], reverse=True)
    d = dcg(relevances)
    id = dcg(ideal_relevances)
    return d / id if id > 0 else 0.0


def compute_mrr(similarities_with_relevance, threshold=0.5):
    """Compute MRR."""
    sorted_by_sim = sorted(similarities_with_relevance, key=lambda x: x[0], reverse=True)
    for i, (_, rel) in enumerate(sorted_by_sim):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


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


def create_multimodal_dataset():
    """Create a multimodal dataset with images and captions."""
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
                'caption': f'a {color_name} {pattern} image',
            })

    return images, metadata


def main():
    parser = argparse.ArgumentParser(description='Multimodal quality benchmark')
    parser.add_argument('--k', type=int, default=10, help='Top-K for metrics')
    args = parser.parse_args()

    print("=" * 70)
    print("Multimodal Quality Benchmark")
    print("=" * 70)
    print()

    # Create dataset
    images, metadata = create_multimodal_dataset()
    print(f"Dataset: {len(images)} images")
    print(f"Colors: {len({m['color'] for m in metadata})}")
    print(f"Patterns: {len({m['pattern'] for m in metadata})}")
    print()

    models = [
        'openai/clip-vit-base-patch32',
        'openai/clip-vit-base-patch32-quantized',
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
        print("--- Image->Image Retrieval ---")
        recalls = []
        ndcgs = []
        mrrs = []
        maps = []

        for i, query_emb in enumerate(embeddings):
            similarities = []
            for j, cand_emb in enumerate(embeddings):
                if i == j:
                    continue
                sim = cosine_similarity(query_emb, cand_emb)
                # Ground truth: same color = relevant
                gt_sim = 1.0 if metadata[i]['color'] == metadata[j]['color'] else 0.0
                similarities.append((sim, gt_sim))

            recalls.append(compute_recall(similarities, args.k))
            ndcgs.append(compute_ndcg(similarities, args.k))
            mrrs.append(compute_mrr(similarities))
            maps.append(compute_map(similarities))

        avg_recall = sum(recalls) / len(recalls)
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        avg_mrr = sum(mrrs) / len(mrrs)
        avg_map = sum(maps) / len(maps)

        print(f"  Recall@{args.k}: {avg_recall:.4f}")
        print(f"  NDCG@{args.k}:   {avg_ndcg:.4f}")
        print(f"  MRR:          {avg_mrr:.4f}")
        print(f"  mAP:          {avg_map:.4f}")
        print()

        model.close()

    # Markdown
    print("## Results (for markdown)")
    print()
    print(f"| Model | Recall@{args.k} | NDCG@{args.k} | MRR | mAP |")
    print("|-------|---------|---------|-----|-----|")


if __name__ == '__main__':
    main()
