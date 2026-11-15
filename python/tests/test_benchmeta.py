"""Unit tests for benchmarks/_benchmeta.py (no network, no timing assertions).

Auteur: David Orel
SPDX-License-Identifier: MIT
"""

import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import _benchmeta as bm


def _run(**over):
    base = {
        "threads": 4,
        "batch": 32,
        "warmup": 50,
        "iterations": 5,
        "format": "onnx",
        "quantization": "fp32",
        "dataset": "synthetic-1000",
    }
    base.update(over)
    return bm.RunParams(**base)


def _complete_machine():
    return {
        "git_commit": "abc1234",
        "git_dirty": False,
        "libembedding_version": "1.2.3",
        "os": "TestOS 1",
        "cpu_model": "Test CPU",
        "physical_cores": 4,
        "logical_cores": 8,
        "ram_mb": 16384,
        "python": "3.11.0",
    }


class TestPercentiles:
    def test_median_odd_and_even(self):
        assert bm.p50([3.0, 1.0, 2.0]) == 2.0
        assert bm.p50([4.0, 1.0, 2.0, 3.0]) == 2.5

    def test_interpolation(self):
        samples = [float(x) for x in range(1, 101)]  # 1..100
        assert bm.percentile(samples, 0) == 1.0
        assert bm.percentile(samples, 100) == 100.0
        assert bm.p95(samples) == pytest.approx(95.05)

    def test_single_sample(self):
        assert bm.p50([7.0]) == 7.0
        assert bm.p95([7.0]) == 7.0

    def test_unsorted_input_not_mutated(self):
        samples = [3.0, 1.0, 2.0]
        bm.p95(samples)
        assert samples == [3.0, 1.0, 2.0]

    def test_empty_and_bad_q(self):
        with pytest.raises(ValueError):
            bm.p50([])
        with pytest.raises(ValueError):
            bm.percentile([1.0], 101)
        with pytest.raises(ValueError):
            bm.percentile([1.0], -1)

    def test_summarize(self):
        out = bm.summarize_latencies([10.0, 20.0, 30.0])
        assert out["n"] == 3
        assert out["min_ms"] == 10.0
        assert out["max_ms"] == 30.0
        assert out["p50_ms"] == 20.0
        assert out["p95_ms"] == pytest.approx(29.0)

    def test_noise_threshold(self):
        assert bm.relative_delta_pct(100.0, 110.0) == pytest.approx(10.0)
        assert bm.is_noise(100.0, 103.0)
        assert bm.is_noise(100.0, 97.0)
        assert not bm.is_noise(100.0, 106.0)
        with pytest.raises(ValueError):
            bm.relative_delta_pct(0.0, 1.0)


class TestMeasurementHelpers:
    def test_timed_load_returns_value_and_nonnegative_ms(self):
        value, ms = bm.timed_load(lambda: "model")
        assert value == "model"
        assert ms >= 0.0

    def test_timed_load_propagates_errors(self):
        def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            bm.timed_load(boom)

    def test_peak_rss_is_positive_float(self):
        peak = bm.peak_rss_mb()
        assert isinstance(peak, float)
        assert peak > 0.0  # a running interpreter always has a resident set


class TestMachineMetadata:
    def test_collect_has_mandatory_keys(self):
        meta = bm.collect_machine_metadata()
        for key in bm.REQUIRED_MACHINE_FIELDS:
            assert key in meta
        assert meta["python"].count(".") >= 1
        assert meta["logical_cores"] >= 0

    def test_git_info_outside_repo_is_unknown_not_guessed(self, tmp_path):
        info = bm.git_info(tmp_path)
        # tmp_path may be inside no repo; either way dirty is never a guess.
        if info["git_commit"] == bm.UNKNOWN:
            assert info["git_dirty"] is None

    def test_git_info_in_repo(self):
        info = bm.git_info()
        assert "git_commit" in info
        assert info["git_dirty"] in (True, False, None)


class TestReport:
    def test_complete_report_has_no_missing_fields(self):
        report = bm.build_report(_complete_machine(), _run())
        assert bm.missing_fields(report) == []

    def test_unknown_and_zero_are_missing(self):
        machine = _complete_machine()
        machine["cpu_model"] = bm.UNKNOWN
        machine["ram_mb"] = 0
        del machine["python"]
        report = bm.build_report(machine, _run(dataset=""))
        assert bm.missing_fields(report) == [
            "machine.cpu_model",
            "machine.ram_mb",
            "machine.python",
            "run.dataset",
        ]

    def test_json_roundtrip_lf_only(self, tmp_path):
        report = bm.build_report(
            _complete_machine(), _run(batch=[8, 32]), results={"a": {"p50_ms": 1.5}}
        )
        path = tmp_path / "r.json"
        bm.write_json(report, path)
        raw = path.read_bytes()
        assert b"\r" not in raw
        assert not raw.startswith(b"\xef\xbb\xbf")
        loaded = json.loads(raw.decode("utf-8"))
        assert loaded["schema_version"] == bm.SCHEMA_VERSION
        assert loaded["run"]["batch"] == [8, 32]
        assert loaded["results"]["a"]["p50_ms"] == 1.5

    def test_markdown_lists_every_run_field(self):
        machine = _complete_machine()
        machine["git_dirty"] = True
        md = bm.to_markdown(bm.build_report(machine, _run()))
        assert md.startswith("## Run metadata")
        assert "| git commit | abc1234 (dirty) |" in md
        for name in bm.REQUIRED_RUN_FIELDS:
            assert f"| {name} |" in md
        assert "Missing mandatory fields" not in md

    def test_markdown_flags_missing_and_escapes_pipes(self):
        machine = _complete_machine()
        machine["cpu_model"] = "A | B"
        machine["ram_mb"] = 0
        md = bm.to_markdown(bm.build_report(machine, _run()))
        assert "A \\| B" in md
        assert "Missing mandatory fields: machine.ram_mb." in md
