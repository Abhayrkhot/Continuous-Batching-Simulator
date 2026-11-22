"""Per-request and per-step results, latency summaries and SLO attainment."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from batchsim.request import Request

FloatArray = NDArray[np.float64]
PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class SLO:
    """Per-request latency objectives (seconds). A request meets the SLO if all hold."""

    ttft: float = math.inf
    tpot: float = math.inf
    e2e: float = math.inf

    def __str__(self) -> str:
        parts = []
        if math.isfinite(self.ttft):
            parts.append(f"TTFT<={self.ttft * 1e3:g}ms")
        if math.isfinite(self.tpot):
            parts.append(f"TPOT<={self.tpot * 1e3:g}ms")
        if math.isfinite(self.e2e):
            parts.append(f"E2E<={self.e2e:g}s")
        return " & ".join(parts) or "none"


def _col(reqs: Sequence[Request], attr: str) -> FloatArray:
    return np.fromiter((getattr(r, attr) for r in reqs), dtype=np.float64, count=len(reqs))


@dataclass
class SimResult:
    """Everything one simulation run produced.

    Request arrays are in workload order. Times are seconds from the workload's start, as
    the engine sees them; ``frontend_latency`` (API server, tokenizer, network) is added to
    client-facing TTFT and end-to-end latency.
    """

    arrival: FloatArray
    prompt_len: FloatArray
    output_len: FloatArray
    first_scheduled: FloatArray
    first_token: FloatArray
    finish: FloatArray
    preemptions: FloatArray
    max_itl: FloatArray
    steps: dict[str, FloatArray]
    frontend_latency: float = 0.0
    itl: list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_requests(
        cls,
        reqs: Sequence[Request],
        steps: dict[str, FloatArray],
        frontend_latency: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> SimResult:
        itl = [r.itl for r in reqs] if reqs and reqs[0].itl is not None else None
        return cls(
            arrival=_col(reqs, "arrival"),
            prompt_len=_col(reqs, "prompt_len"),
            output_len=_col(reqs, "output_len"),
            first_scheduled=_col(reqs, "first_scheduled"),
            first_token=_col(reqs, "first_token"),
            finish=_col(reqs, "finish"),
            preemptions=_col(reqs, "num_preemptions"),
            max_itl=_col(reqs, "max_itl"),
            steps=steps,
            frontend_latency=frontend_latency,
            itl=itl,  # type: ignore[arg-type]
            meta=meta or {},
        )

    # ------------------------------------------------------------------ per request

    def __len__(self) -> int:
        return len(self.arrival)

    @property
    def ttft(self) -> FloatArray:
        return self.first_token - self.arrival + self.frontend_latency

    @property
    def e2e(self) -> FloatArray:
        return self.finish - self.arrival + self.frontend_latency

    @property
    def queue_delay(self) -> FloatArray:
        return self.first_scheduled - self.arrival

    @property
    def tpot(self) -> FloatArray:
        """Mean time per output token after the first (NaN for single-token outputs)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (self.finish - self.first_token) / (self.output_len - 1)
        out[self.output_len <= 1] = np.nan
        return out

    def mask(self, warmup: float = 0.0, cooldown: float = 0.0) -> NDArray[np.bool_]:
        """Requests that arrived after ``warmup`` s and before ``cooldown`` s from the end."""
        if len(self) == 0:
            return np.zeros(0, dtype=bool)
        t0, t1 = float(np.min(self.arrival)), float(np.max(self.arrival))
        mask: NDArray[np.bool_] = (self.arrival >= t0 + warmup) & (self.arrival <= t1 - cooldown)
        return mask

    def slo_attainment(self, slo: SLO, mask: NDArray[np.bool_] | None = None) -> float:
        m = np.ones(len(self), dtype=bool) if mask is None else mask
        if not m.any():
            return math.nan
        tpot = np.nan_to_num(self.tpot, nan=0.0)
        ok = (self.ttft <= slo.ttft) & (tpot <= slo.tpot) & (self.e2e <= slo.e2e)
        return float(ok[m].mean())

    # ------------------------------------------------------------------ summary

    def summary(
        self, warmup: float = 0.0, cooldown: float = 0.0, slo: SLO | None = None
    ) -> dict[str, float]:
        m = self.mask(warmup, cooldown)
        out: dict[str, float] = {"requests": float(m.sum())}
        if not m.any():
            return out
        span = float(np.max(self.finish) - np.min(self.arrival))
        out["duration_s"] = span
        out["throughput_rps"] = len(self) / span if span > 0 else math.nan
        out["output_tok_s"] = float(self.output_len.sum()) / span if span > 0 else math.nan
        out["total_tok_s"] = (
            float(self.output_len.sum() + self.prompt_len.sum()) / span if span > 0 else math.nan
        )
        for name, arr in (("ttft", self.ttft), ("tpot", self.tpot), ("e2e", self.e2e)):
            vals = arr[m]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            out[f"{name}_mean"] = float(vals.mean())
            for p, v in zip(PERCENTILES, np.percentile(vals, PERCENTILES)):
                out[f"{name}_p{p}"] = float(v)
        out["preemptions"] = float(self.preemptions.sum())
        if self.steps and len(self.steps.get("start", ())):
            st = self.steps
            out["steps"] = float(len(st["start"]))
            weights = st["dur"]
            out["mean_batch_seqs"] = float(np.average(st["num_seqs"], weights=weights))
            out["mean_batch_tokens"] = float(np.average(st["num_tokens"], weights=weights))
            out["peak_kv_usage"] = float(np.max(st["kv_usage"]))
            busy = float(weights.sum())
            out["busy_frac"] = busy / span if span > 0 else math.nan
        if slo is not None:
            out["slo_attainment"] = self.slo_attainment(slo, m)
        return out

    def to_json(self, path: str | Path, **extra: Any) -> None:
        Path(path).write_text(
            json.dumps({"meta": self.meta, "summary": self.summary(), **extra}, indent=2) + "\n"
        )


def weighted_percentile(values: FloatArray, weights: FloatArray, q: Sequence[float]) -> FloatArray:
    """Percentiles of ``values`` where each value counts ``weights`` times."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return np.full(len(q), np.nan)
    targets = np.asarray(q, dtype=np.float64) / 100.0 * cum[-1]
    idx = np.searchsorted(cum, targets, side="left").clip(0, len(v) - 1)
    out: FloatArray = v[idx]
    return out
