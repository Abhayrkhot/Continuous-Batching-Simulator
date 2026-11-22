"""Fit a :class:`LinearStepModel` to iterations recorded inside a real vLLM engine.

Input is the JSONL written by ``bench/batchsim_vllm_probe.py``: one line per engine
iteration with its batch shape and the wall-clock times of CUDA events recorded before the
forward pass (``g0``) and after sampling (``g1``).

The target is the iteration's *period*: from the end of the previous iteration to the end
of this one when the engine ran them back to back, so host gaps between iterations (which
grow with batch size) are charged to the iteration that caused them. After an idle gap the
period is just ``g1 - g0``.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from batchsim.cost import VLLM_CAPTURE_SIZES, BatchStats, LinearStepModel

DEFAULT_KNOTS = (256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class StepRecord:
    index: int
    cpu: float
    g0: float
    g1: float
    stats: BatchStats


def stats_from_probe(rec: dict[str, Any]) -> BatchStats:
    s = BatchStats()
    s.num_tokens = int(rec["T"])
    s.num_seqs = int(rec["S"])
    s.num_decode = int(rec["D"])
    s.decode_ctx = float(rec["Dc"]) + s.num_decode  # each decode query reads context + itself
    for n, c in rec["p"]:
        s.num_prefill += 1
        s.prefill_tokens += n
        s.prefill_attn += n * c + n * (n + 1) / 2
    return s


def open_text(path: str | Path) -> IO[str]:
    """Open ``path``, or ``path.gz`` if only the compressed copy exists (as in results/)."""
    path = Path(path)
    if path.suffix == ".gz" or (not path.exists() and path.with_name(path.name + ".gz").exists()):
        gz = path if path.suffix == ".gz" else path.with_name(path.name + ".gz")
        return gzip.open(gz, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def load_probe(paths: str | Path | Iterable[str | Path]) -> list[StepRecord]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for path in paths:
        with open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                out.append(StepRecord(r["i"], r["cpu"], r["g0"], r["g1"], stats_from_probe(r)))
    out.sort(key=lambda r: r.g0)
    return out


def periods(steps: Sequence[StepRecord], idle_gap: float = 2e-3) -> NDArray[np.float64]:
    """Per-iteration period in seconds (see module docstring)."""
    y = np.empty(len(steps))
    prev_end = -math.inf
    for k, s in enumerate(steps):
        gap = s.g0 - prev_end
        y[k] = s.g1 - prev_end if gap < idle_gap else s.g1 - s.g0
        prev_end = s.g1
    return y


def nnls(a: NDArray[np.float64], b: NDArray[np.float64], tol: float = 1e-12) -> NDArray[np.float64]:
    """Non-negative least squares, ``min ||a x - b||`` s.t. ``x >= 0`` (Lawson-Hanson)."""
    n = a.shape[1]
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1.0
    a = a / scale
    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    w = a.T @ (b - a @ x)
    for _ in range(30 * n):
        if passive.all() or w[~passive].max() <= tol:
            break
        j = int(np.argmax(np.where(passive, -np.inf, w)))
        passive[j] = True
        while True:
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
            if (z[passive] > tol).all():
                break
            neg = passive & (z <= tol)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            passive &= x > tol
            if not passive.any():
                z = np.zeros(n)
                break
        x = z
        w = a.T @ (b - a @ x)
    return x / scale


@dataclass(frozen=True)
class FitReport:
    n_train: int
    n_dropped: int
    mape: float
    median_ape: float
    p90_ape: float
    p99_ape: float
    bias: float  # mean (pred - y) / y
    r2: float

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def evaluate(
    model: LinearStepModel, stats: Sequence[BatchStats], y: NDArray[np.float64]
) -> FitReport:
    pred = np.array([model.step_time(s) for s in stats])
    rel = (pred - y) / y
    ape = np.abs(rel)
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return FitReport(
        n_train=len(y),
        n_dropped=0,
        mape=float(ape.mean()),
        median_ape=float(np.median(ape)),
        p90_ape=float(np.percentile(ape, 90)),
        p99_ape=float(np.percentile(ape, 99)),
        bias=float(rel.mean()),
        r2=1 - ss_res / ss_tot if ss_tot > 0 else math.nan,
    )


def fit_linear(
    stats: Sequence[BatchStats],
    y: NDArray[np.float64],
    knots: Sequence[int] | None = None,
    capture_sizes: Sequence[int] = VLLM_CAPTURE_SIZES,
    trim_quantile: float = 0.995,
    meta: dict[str, Any] | None = None,
) -> tuple[LinearStepModel, FitReport]:
    """Fit by NNLS on relative error, then refit without the worst residuals.

    Relative error weighting stops the few long prefill iterations from dominating the
    many short decode ones. Trimming removes one-off stalls (allocator growth, a
    first-time kernel autotune) that no batch-shape model should explain.
    """
    if len(stats) != len(y) or len(y) == 0:
        raise ValueError("need one target per step and at least one step")
    max_t = max(s.num_tokens for s in stats)
    knots = tuple(k for k in (knots if knots is not None else DEFAULT_KNOTS) if k < max_t)
    x = np.array([LinearStepModel.features(s, knots, capture_sizes) for s in stats])
    names = LinearStepModel.feature_names(knots)
    keep = np.ones(len(y), dtype=bool)

    def solve(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        w = 1.0 / y[mask]
        return nnls(x[mask] * w[:, None], np.ones(int(mask.sum())))

    coef = solve(keep)
    rel = np.abs(x @ coef - y) / y
    if 0 < trim_quantile < 1:
        keep = rel <= np.quantile(rel, trim_quantile)
        coef = solve(keep)
    model = LinearStepModel(
        coef=dict(zip(names, (float(c) for c in coef))),
        knots=knots,
        capture_sizes=tuple(capture_sizes),
        meta=meta or {},
    )
    rep = evaluate(model, [s for s, k in zip(stats, keep) if k], y[keep])
    rep = FitReport(**{**rep.to_dict(), "n_dropped": int((~keep).sum())})  # type: ignore[arg-type]
    return model, rep


def calibrate_from_probe(
    paths: str | Path | Iterable[str | Path],
    idle_gap: float = 2e-3,
    holdout: float = 0.2,
    seed: int = 0,
    knots: Sequence[int] | None = None,
    capture_sizes: Sequence[int] = VLLM_CAPTURE_SIZES,
    meta: dict[str, Any] | None = None,
) -> tuple[LinearStepModel, dict[str, Any]]:
    """Fit on a random ``1 - holdout`` of the iterations, report error on the rest, then
    refit on all of them for the returned model."""
    steps = load_probe(paths)
    y = periods(steps, idle_gap)
    stats = [s.stats for s in steps]
    rng = np.random.default_rng(seed)
    test = rng.random(len(y)) < holdout
    train_idx = np.flatnonzero(~test)
    model_tr, rep_tr = fit_linear([stats[i] for i in train_idx], y[~test], knots, capture_sizes)
    rep_te = evaluate(model_tr, [stats[i] for i in np.flatnonzero(test)], y[test])
    model, rep_all = fit_linear(stats, y, knots, capture_sizes, meta=meta)
    report = {
        "steps": len(y),
        "train": rep_tr.to_dict(),
        "holdout": rep_te.to_dict(),
        "all": rep_all.to_dict(),
        "idle_gap_s": idle_gap,
    }
    model.meta = {**(meta or {}), "fit": report}
    return model, report
