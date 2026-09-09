"""
Extended edge cases test for libembedding.
Tests boundary conditions and error handling.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding, TextEmbeddingPool
import numpy as np

def test_edge_cases():
    """Test various edge cases."""
    print("=== Extended Edge Cases ===\n")

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=1,
        offline=True,
        show_download_progress=False,
    )

    cases = [
        # Basic edge cases
        ("Empty string", ""),
        ("Single char", "a"),
        ("Whitespace only", "   "),
        ("Tab only", "\t"),
        ("Newline only", "\n"),
        ("Carriage return", "\r\n"),

        # Special characters
        ("Numbers", "12345 67890"),
        ("Special chars", "!@#$%^&_+-=[]{}|;':\",./<>?"),
        ("Punctuation only", "...,,,!!!???"),
        ("Math symbols", "∑∏∫∂√≈≠≤≥"),
        ("Currency", "€£¥₹₽₩"),

        # Unicode and emoji
        ("Unicode emoji", "Hello 😀 🎉 🌍 world"),
        ("Skin tone emoji", "👋🏻👋🏿"),
        ("Flag emoji", "🇫🇷🇩🇪🇯🇵🇺🇸"),
        ("CJK mixed", "Hello 世界 안녕 こんにちは"),
        ("Right-to-left", "Hello مرحبا world"),
        ("Zero-width space", "Hello​world"),
        ("Combining chars", "é̈̃"),

        # Long texts
        ("Repeated word", "test " * 100),
        ("Very long word", "a" * 1000),
        ("Long text (512+ tokens)", "word " * 512),

        # Mixed content
        ("Code snippet", "def foo(): return 42\nbar()"),
        ("JSON", '{"key": "value", "num": 123}'),
        ("URL", "https://example.com/path?q=1&b=2"),
        ("Email", "user@example.com"),
        ("HTML", "<div class='test'>Hello</div>"),

        # Boundary tokens
        ("16 tokens", "word " * 16),
        ("64 tokens", "word " * 64),
        ("128 tokens", "word " * 128),
        ("256 tokens", "word " * 256),
        ("512 tokens", "word " * 512),
    ]

    passed = 0
    failed = 0

    for name, text in cases:
        try:
            result = model.embed([text])
            if result.shape[0] == 1:
                # Check for NaN/Inf
                has_nan = np.any(np.isnan(result)) or np.any(np.isinf(result))
                if has_nan:
                    print(f"  {name:<30} FAIL (NaN/Inf)")
                    failed += 1
                else:
                    print(f"  {name:<30} OK (dim={result.shape[1]})")
                    passed += 1
            else:
                print(f"  {name:<30} FAIL (n={result.shape[0]})")
                failed += 1
        except Exception as e:
            print(f"  {name:<30} ERROR ({type(e).__name__}: {e})")
            failed += 1

    model.close()
    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed, failed


def test_large_batch():
    """Test with a large batch of texts."""
    print("\n=== Large Batch Test ===\n")

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=4,
        batch_size=128,
        offline=True,
        show_download_progress=False,
    )

    # Generate 1000 texts
    texts = [f"This is test sentence number {i} for batch processing." for i in range(1000)]

    print(f"  Embedding {len(texts)} texts...")

    import time
    t0 = time.perf_counter()
    result = model.embed(texts)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    docs_per_sec = len(texts) / elapsed

    print(f"  Result: {result.shape[0]} embeddings in {elapsed*1000:.0f} ms")
    print(f"  Throughput: {docs_per_sec:.0f} docs/s")

    # Verify all embeddings are valid
    nan_count = np.any(np.isnan(result)) or np.any(np.isinf(result))
    print(f"  NaN/Inf detected: {nan_count}")

    model.close()
    return docs_per_sec


def test_pool_edge_cases():
    """Test TextEmbeddingPool with edge cases."""
    print("\n=== Pool Edge Cases ===\n")

    pool = TextEmbeddingPool(
        "sentence-transformers/all-MiniLM-L6-v2",
        workers=4,
        threads_per_worker=1,
        batch_size=64,
        offline=True,
        show_download_progress=False,
    )

    # Test with various batch sizes
    for n_texts in [1, 2, 4, 8, 16, 37, 100]:
        texts = [f"Text {i}" for i in range(n_texts)]

        import time
        t0 = time.perf_counter()
        result = pool.embed(texts)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        docs_per_sec = n_texts / elapsed if elapsed > 0 else float('inf')

        print(f"  {n_texts:4d} texts: {elapsed*1000:.1f} ms = {docs_per_sec:.0f} docs/s")

    pool.close()
    return True


def main():
    passed, failed = test_edge_cases()
    test_large_batch()
    test_pool_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
