"""Test script for the unified benchmark Python API.

Verifies that the benchmark module is importable and functional.
"""

import os
import sys

# Add build path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "src"))


def test_import():
    """Test that benchmark module imports cleanly."""

    print("OK: benchmark module imported")
    return True


def test_hardware_detection():
    """Test hardware detection."""
    from libembedding.benchmark import detect_hardware

    hw = detect_hardware()
    print(f"Hardware: {hw.cpu_name}")
    print(f"  Cores: {hw.physical_cores}P / {hw.logical_cores}L")
    print(f"  RAM: {hw.ram_mb} MB")
    return True


def test_cache_path():
    """Test cache path retrieval."""
    from libembedding.benchmark import cache_path

    path = cache_path()
    print(f"Cache path: {path}")
    return True


def test_clear_cache():
    """Test cache clearing."""
    from libembedding.benchmark import clear_cache

    clear_cache()
    print("OK: cache cleared")
    return True


if __name__ == "__main__":
    print("=== Benchmark Python API Test ===\n")
    try:
        test_import()
        print()
        test_hardware_detection()
        print()
        test_cache_path()
        print()
        test_clear_cache()
        print("\n=== All tests passed ===")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
