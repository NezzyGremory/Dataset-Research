"""Benchmark runtime dan peak RSS pada beberapa ukuran DataFrame."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from statistics import median

import numpy as np
import pandas as pd

from app.analyzer.correlations import CorrelationAnalyzer
from app.analyzer.duplicates import DuplicateAnalyzer
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.outliers import OutlierAnalyzer
from app.analyzer.profiler import DatasetProfiler
from evaluation.metrics import bootstrap_ci, wilcoxon_signed_rank


def _rss_bytes() -> int | None:
    """Ambil RSS proses menggunakan API OS tanpa dependensi tambahan."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class _PeakSampler:
    """Sampling RSS selama satu blok benchmark."""

    def __enter__(self):
        self.start = _rss_bytes()
        self.peak = self.start
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.01):
            current = _rss_bytes()
            if current is not None and (self.peak is None or current > self.peak):
                self.peak = current

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        current = _rss_bytes()
        if current is not None and (self.peak is None or current > self.peak):
            self.peak = current


def _dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"feature_{i}": rng.normal(size=rows) for i in range(6)})


def _measure(function, repeats: int) -> tuple[list[float], list[int | None]]:
    times, peaks = [], []
    for _ in range(repeats):
        with _PeakSampler() as sampler:
            start = time.perf_counter()
            function()
            times.append((time.perf_counter() - start) * 1000)
        peaks.append(max(0, sampler.peak - sampler.start) if sampler.peak is not None and sampler.start is not None else None)
    return times, peaks


def run_performance(quick: bool = False) -> dict:
    """Ukur analyzer dan operasi pandas pembanding dengan input terkunci."""
    sizes = [1000] if quick else [1000, 10_000, 100_000, 500_000]
    repeats = 5
    results = []
    analyzers = {
        "profile": (DatasetProfiler().profile, lambda frame: {"shape": frame.shape, "memory_bytes": int(frame.memory_usage(deep=True).sum())}),
        "missing": (MissingValueAnalyzer().analyze, lambda frame: frame.isna().sum().to_dict()),
        "duplicates": (DuplicateAnalyzer().analyze, lambda frame: int(frame.duplicated().sum())),
        "outliers": (OutlierAnalyzer().analyze, lambda frame: [frame[column].quantile([.25, .75]).to_list() for column in frame.select_dtypes(include="number")]),
        "correlations": (CorrelationAnalyzer().analyze, lambda frame: frame.select_dtypes(include="number").corr(method="pearson")),
    }
    for size in sizes:
        frame = _dataset(size)
        for stage, (application, baseline) in analyzers.items():
            app_times, app_peaks = _measure(lambda: application(frame), repeats)
            base_times, base_peaks = _measure(lambda: baseline(frame), repeats)
            app_ci = bootstrap_ci(app_times, n_resamples=500, seed=42)
            paired = wilcoxon_signed_rank(app_times, base_times) if repeats > 1 else None
            results.append({"rows": size, "stage": stage,
                            "dataframe_deep_memory_bytes": int(frame.memory_usage(deep=True).sum()),
                            "application_time_ms": app_times, "application_median_ms": median(app_times),
                            "application_time_ci95": app_ci, "application_peak_rss_delta_bytes": app_peaks,
                            "baseline_time_ms": base_times, "baseline_median_ms": median(base_times),
                            "baseline_peak_rss_delta_bytes": base_peaks, "paired_wilcoxon": paired})
    return {"sizes": sizes, "repeats": repeats, "seed": 42,
            "memory_note": "Peak RSS adalah delta dari sampling proses 10 ms; alokasi sangat singkat dapat terlewat.",
            "results": results}
