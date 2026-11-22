"""Workloads: synthetic generators, the Azure LLM inference traces, and CSV exchange.

A :class:`Workload` is an immutable, arrival-ordered list of :class:`RequestSpec`. The same
CSV file drives both the simulator and the real-engine benchmark client
(``bench/replay.py``), so both see byte-identical arrival times and lengths.
"""

from __future__ import annotations

import csv
import itertools
import math
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from batchsim.request import RequestSpec

AZURE_TRACE_URLS = {
    "conv": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
    "code": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv",
}


@dataclass(frozen=True)
class Workload:
    specs: tuple[RequestSpec, ...]
    name: str = "workload"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrivals = [s.arrival for s in self.specs]
        if any(b < a for a, b in itertools.pairwise(arrivals)):
            raise ValueError("workload specs must be sorted by arrival time")

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Any:
        return iter(self.specs)

    @property
    def duration(self) -> float:
        return self.specs[-1].arrival - self.specs[0].arrival if len(self.specs) > 1 else 0.0

    @property
    def rate(self) -> float:
        """Mean arrival rate in requests per second."""
        if len(self.specs) < 2 or self.duration == 0:
            return math.inf
        return (len(self.specs) - 1) / self.duration

    @property
    def prompt_lens(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.array([s.prompt_len for s in self.specs], dtype=np.int64)

    @property
    def output_lens(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.array([s.output_len for s in self.specs], dtype=np.int64)

    def with_specs(self, specs: Iterable[RequestSpec], **meta: Any) -> Workload:
        return Workload(tuple(specs), self.name, {**self.meta, **meta})

    def scale_to_rate(self, rate: float) -> Workload:
        """Stretch or compress time so the mean arrival rate is ``rate``.

        Time scaling keeps the trace's burstiness (its inter-arrival coefficient of
        variation) while changing the load.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        factor = self.rate / rate
        t0 = self.specs[0].arrival if self.specs else 0.0
        return self.with_specs(
            (
                RequestSpec((s.arrival - t0) * factor, s.prompt_len, s.output_len, s.max_tokens)
                for s in self.specs
            ),
            rate=rate,
        )

    def head(self, n: int) -> Workload:
        return self.with_specs(self.specs[:n])

    def window(self, start: float, end: float) -> Workload:
        """Requests arriving in ``[start, end)``, re-based to start at zero."""
        picked = [s for s in self.specs if start <= s.arrival < end]
        return self.with_specs(
            RequestSpec(s.arrival - start, s.prompt_len, s.output_len, s.max_tokens) for s in picked
        )

    def clip(self, max_model_len: int, max_output: int | None = None) -> Workload:
        """Clip lengths so every request fits ``max_model_len`` (prompt + output)."""
        out = []
        for s in self.specs:
            o = s.output_len if max_output is None else min(s.output_len, max_output)
            o = min(o, max_model_len - 1)
            p = min(s.prompt_len, max_model_len - o)
            mt = None if s.max_tokens is None else max(o, min(s.max_tokens, max_model_len - p))
            out.append(RequestSpec(s.arrival, p, o, mt))
        return self.with_specs(out, max_model_len=max_model_len)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arrival", "prompt_len", "output_len", "max_tokens"])
            for s in self.specs:
                w.writerow([f"{s.arrival:.6f}", s.prompt_len, s.output_len, s.max_tokens or ""])

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> Workload:
        path = Path(path)
        specs = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                mt = row.get("max_tokens") or None
                specs.append(
                    RequestSpec(
                        float(row["arrival"]),
                        int(row["prompt_len"]),
                        int(row["output_len"]),
                        int(mt) if mt else None,
                    )
                )
        specs.sort(key=lambda s: s.arrival)
        return cls(tuple(specs), name or path.stem)


# --------------------------------------------------------------------------- lengths


class LengthDist(Protocol):
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.int64]]: ...


@dataclass(frozen=True)
class Fixed:
    value: int

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.full(n, self.value, dtype=np.int64)


@dataclass(frozen=True)
class Uniform:
    low: int
    high: int  # inclusive

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        return rng.integers(self.low, self.high + 1, size=n, dtype=np.int64)


@dataclass(frozen=True)
class LogNormal:
    """Log-normal with the given mean and coefficient of variation, clipped to [low, high]."""

    mean: float
    cv: float
    low: int = 1
    high: int = 1 << 30

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        sigma2 = math.log1p(self.cv**2)
        mu = math.log(self.mean) - sigma2 / 2
        x = rng.lognormal(mu, math.sqrt(sigma2), size=n)
        return np.clip(np.rint(x), self.low, self.high).astype(np.int64)


@dataclass(frozen=True)
class Empirical:
    values: tuple[int, ...]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        return rng.choice(np.asarray(self.values, dtype=np.int64), size=n, replace=True)


def arrivals(
    n: int, rate: float, rng: np.random.Generator, process: str = "poisson", cv: float = 1.0
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Arrival times of ``n`` requests at mean ``rate`` req/s.

    ``poisson`` has exponential gaps; ``gamma`` has gamma gaps with coefficient of
    variation ``cv`` (cv > 1 is burstier than Poisson); ``uniform`` is evenly spaced.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if process == "poisson":
        gaps = rng.exponential(1.0 / rate, size=n)
    elif process == "gamma":
        shape = 1.0 / cv**2
        gaps = rng.gamma(shape, 1.0 / (rate * shape), size=n)
    elif process == "uniform":
        gaps = np.full(n, 1.0 / rate)
    else:
        raise ValueError(f"unknown arrival process {process!r}")
    gaps[0] = 0.0
    return np.cumsum(gaps)


def synthetic(
    n: int,
    rate: float,
    prompt: LengthDist,
    output: LengthDist,
    *,
    process: str = "poisson",
    cv: float = 1.0,
    seed: int = 0,
    max_tokens: int | None = None,
    name: str = "synthetic",
) -> Workload:
    rng = np.random.default_rng(seed)
    t = arrivals(n, rate, rng, process, cv)
    p = prompt.sample(rng, n)
    o = output.sample(rng, n)
    specs = tuple(
        RequestSpec(
            float(t[i]),
            int(p[i]),
            int(o[i]),
            None if max_tokens is None else max(max_tokens, int(o[i])),
        )
        for i in range(n)
    )
    return Workload(specs, name, {"rate": rate, "process": process, "cv": cv, "seed": seed})


def from_lengths(
    prompt_lens: Sequence[int],
    output_lens: Sequence[int],
    arrival_times: Sequence[float] | None = None,
    name: str = "workload",
) -> Workload:
    """Build a workload from parallel arrays; all requests arrive at t=0 by default."""
    times = arrival_times if arrival_times is not None else [0.0] * len(prompt_lens)
    specs = sorted(
        (RequestSpec(float(t), int(p), int(o)) for t, p, o in zip(times, prompt_lens, output_lens)),
        key=lambda s: s.arrival,
    )
    return Workload(tuple(specs), name)


# --------------------------------------------------------------------------- Azure traces


def _parse_ts(text: str) -> datetime:
    text = text.strip()
    if "." in text:  # the trace has 7 fractional digits; datetime parses at most 6
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(text)


def load_azure_trace(path: str | Path, name: str | None = None) -> Workload:
    """Load an Azure LLM inference trace (Patel et al., Splitwise, ISCA 2024).

    Columns: ``TIMESTAMP, ContextTokens, GeneratedTokens``. Requests with zero generated
    tokens are given one (every request in the simulator produces at least one token).
    """
    path = Path(path)
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    _parse_ts(row["TIMESTAMP"]),
                    int(row["ContextTokens"]),
                    int(row["GeneratedTokens"]),
                )
            )
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    specs = tuple(RequestSpec((ts - t0).total_seconds(), max(1, p), max(1, o)) for ts, p, o in rows)
    return Workload(specs, name or path.stem, {"source": "azure", "file": path.name})


def fetch_azure_trace(kind: str, dest_dir: str | Path = "data") -> Path:
    """Download the Azure ``conv`` or ``code`` trace (about 1 MB each) if not present."""
    if kind not in AZURE_TRACE_URLS:
        raise ValueError(f"kind must be one of {sorted(AZURE_TRACE_URLS)}")
    dest = Path(dest_dir) / f"AzureLLMInferenceTrace_{kind}.csv"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(AZURE_TRACE_URLS[kind], tmp)
        tmp.rename(dest)
    return dest


def azure(kind: str, data_dir: str | Path = "data") -> Workload:
    return load_azure_trace(fetch_azure_trace(kind, data_dir), name=f"azure-{kind}")
