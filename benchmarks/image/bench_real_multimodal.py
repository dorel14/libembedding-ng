"""
Real image multimodal benchmark: Text→Image retrieval.
Uses CLIP text encoder + vision encoder with real images.
"""
import argparse
import glob
import math
import os
import sys

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import ImageEmbedding, TextEmbedding


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
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


def compute_mrr(similarities, threshold=0.5):
    """Compute MRR."""
    sorted_by_sim = sorted(similarities, key=lambda x: x[0], reverse=True)
    for i, (_, rel) in enumerate(sorted_by_sim):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_map(similarities, threshold=0.5):
    """Compute Average Precision."""
    sorted_by_sim = sorted(similarities, key=lambda x: x[0], reverse=True)
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


def load_images_from_dir(image_dir, max_images=50):
    """Load images from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

    # Filter out very small files (< 1KB) and very large files (> 5MB)
    filtered = []
    for p in image_paths:
        size = os.path.getsize(p)
        if 1000 < size < 5 * 1024 * 1024:
            filtered.append(p)

    return filtered[:max_images]


def main():
    parser = argparse.ArgumentParser(description='Real image multimodal benchmark')
    parser.add_argument('--image-dir', default=None, help='Directory with images')
    parser.add_argument('--k', type=int, default=5, help='Top-K for metrics')
    parser.add_argument('--iterations', type=int, default=3, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 70)
    print("Real Image Multimodal Benchmark: Text->Image")
    print("=" * 70)
    print()

    # Find images
    if args.image_dir:
        image_dir = args.image_dir
    else:
        # Default to user's Pictures folder
        image_dir = os.path.expanduser('~/OneDrive/Images')

    image_paths = load_images_from_dir(image_dir, max_images=30)
    print(f"Found {len(image_paths)} images in {image_dir}")
    if len(image_paths) == 0:
        print("No images found!")
        return

    # Show some image names
    for i, p in enumerate(image_paths[:5]):
        print(f"  [{i}] {os.path.basename(p)} ({os.path.getsize(p) / 1024:.0f} KB)")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")
    print()

    # Define text queries with expected relevant images
    # For this benchmark, we'll use generic queries and measure self-consistency
    queries = [
        "a photo of a person",
        "a screenshot",
        "a document",
        "a face",
        "text on screen",
    ]

    models = [
        ('openai/clip-vit-base-patch32', 'openai/clip-vit-base-patch32', 'FP32'),
        ('openai/clip-vit-base-patch32', 'openai/clip-vit-base-patch32-quantized', 'INT8'),
    ]

    for text_model, vision_model, label in models:
        print(f"{'='*70}")
        print(f"Model: {label} (text={text_model}, vision={vision_model})")
        print(f"{'='*70}")
        print()

        # Load encoders
        print("Loading text encoder...")
        text_encoder = TextEmbedding(text_model, offline=True, show_download_progress=False, threads=4)
        print("Loading vision encoder...")
        vision_encoder = ImageEmbedding(vision_model, offline=True, show_download_progress=False, threads=4)

        # Embed all images
        print(f"Embedding {len(image_paths)} images...")
        image_embs = []
        valid_paths = []
        for i, path in enumerate(image_paths):
            try:
                with open(path, 'rb') as f:
                    img_bytes = f.read()
                emb = vision_encoder.embed_bytes([img_bytes], batch_size=1)
                image_embs.append(emb[0])
                valid_paths.append(path)
            except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
                print(f"  Failed to embed {os.path.basename(path)}: {e}")

        print(f"Successfully embedded {len(image_embs)} images")
        print()

        # Text->Image retrieval
        print("--- Text->Image Retrieval ---")

        for query in queries:
            # Encode text query
            text_emb = text_encoder.embed([query])[0]

            # Compute similarities to all images
            similarities = []
            for j, img_emb in enumerate(image_embs):
                sim = cosine_similarity(text_emb, img_emb)
                # Ground truth: we don't have manual labels, so we use
                # a simple heuristic: images with similarity > median are "relevant"
                similarities.append((sim, 0))  # placeholder

            # For this benchmark, we measure self-consistency:
            # how well does the top-k match the query?
            # Since we don't have ground truth labels, we report raw similarities
            sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)
            top_k_sims = [s for s, _ in sorted_sims[:args.k]]
            avg_top_k = sum(top_k_sims) / len(top_k_sims) if top_k_sims else 0

            print(f"  Query: '{query}'")
            print(f"    Top-{args.k} avg similarity: {avg_top_k:.4f}")
            print(f"    Top match: {os.path.basename(valid_paths[sorted_sims[0][1]] if sorted_sims else 0)}")

        print()

        # Image->Image retrieval (consistency check)
        print("--- Image->Image Retrieval (self-consistency) ---")
        recalls = []
        ndcgs = []

        for i, query_emb in enumerate(image_embs[:10]):  # Use first 10 as queries
            similarities = []
            for j, cand_emb in enumerate(image_embs):
                if i == j:
                    continue
                sim = cosine_similarity(query_emb, cand_emb)
                # Ground truth: same image = most relevant (but we skip self)
                gt = 1.0 if os.path.basename(valid_paths[i]) == os.path.basename(valid_paths[j]) else 0.0
                similarities.append((sim, gt))

            recalls.append(compute_recall(similarities, args.k))
            ndcgs.append(compute_ndcg(similarities, args.k))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

        print(f"  Recall@{args.k}: {avg_recall:.4f}")
        print(f"  NDCG@{args.k}:   {avg_ndcg:.4f}")
        print()

        text_encoder.close()
        vision_encoder.close()

    print("=" * 70)
    print("NOTES:")
    print("- Text->Image: No ground truth labels, showing raw similarities")
    print("- Image->Image: Self-consistency check (same image = relevant)")
    print("- For proper evaluation, manual labels are needed")
    print("=" * 70)


if __name__ == '__main__':
    main()
