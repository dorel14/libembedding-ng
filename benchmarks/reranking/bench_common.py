"""
Reranking benchmark utilities - shared helpers for all reranking benchmarks.
"""
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))



def get_rss_mb():
    """Get current process RSS in MB (cross-platform)."""
    if platform.system() == 'Windows':
        import ctypes
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 * 1024)
    elif platform.system() == 'Darwin':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def generate_documents(n, seed=42):
    """Generate n synthetic documents of varying lengths."""
    import random
    random.seed(seed)
    base_docs = [
        "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The Python programming language was created by Guido van Rossum in 1991.",
        "Quantum computing leverages quantum mechanical phenomena to perform computation.",
        "The Great Wall of China is a series of fortifications built over centuries.",
        "Natural language processing enables computers to understand human language.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "Shakespeare wrote 37 plays during his lifetime in Elizabethan England.",
        "CRISPR gene editing allows scientists to modify DNA sequences with precision.",
        "Electric vehicles are becoming more popular as battery technology improves.",
        "The theory of relativity changed our understanding of space, time, and gravity.",
        "Blockchain technology enables decentralized and transparent digital transactions.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Internet of Things connects billions of devices to exchange data.",
        "Antibiotics revolutionized medicine by treating bacterial infections effectively.",
        "Virtual reality creates immersive digital environments for users to explore.",
    ]
    docs = []
    for i in range(n):
        doc = base_docs[i % len(base_docs)]
        repeat = (i % 3) + 1
        docs.append((doc + " ") * repeat)
    return docs


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def p95(values):
    s = sorted(values)
    idx = int(0.95 * len(s)) - 1
    return s[max(0, min(idx, len(s) - 1))]


def benchmark_rerank(reranker, query, docs, batch_size, warmup=1, iterations=10):
    """Run rerank benchmark and return timing statistics."""
    for _ in range(warmup):
        reranker.rerank(query, docs, batch_size=batch_size)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        reranker.rerank(query, docs, batch_size=batch_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        'median_ms': median(times),
        'p95_ms': p95(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'mean_ms': sum(times) / len(times),
    }
