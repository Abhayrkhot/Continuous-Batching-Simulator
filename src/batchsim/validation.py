"""Compare simulator predictions with measurements from a real vLLM server.

A run directory (written by ``bench/explorer.sbatch``) holds:

``server.log``    vLLM's log; the KV cache size is read from it
``meta.json``     model, vLLM arguments, Slurm job
``runs.jsonl``    wall-clock window of each replayed workload
``<run>.jsonl``   per-request client records from ``bench/replay.py``
``steps.jsonl``   per-iteration probe records (see ``batchsim.calibrate``)

For each replayed workload the simulator is fed the *actual* send times recorded by the
client (not the intended ones), with the server's exact limits and KV block count.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from batchsim.calibrate import StepRecord, exists, load_probe, open_text, periods
from batchsim.cost import StepTimeModel
from batchsim.engine import EngineConfig, simulate
from batchsim.metrics import PERCENTILES, SimResult
from batchsim.request import RequestSpec
from batchsim.schedulers import SchedulerConfig
from batchsim.workload import Workload

# vLLM 0.34 OpenAI-server defaults on GPUs with >= 70 GiB that are not A100s (H100/H200).
H200_DEFAULTS = {"max_num_batched_tokens": 8192, "max_num_seqs": 1024}
METRICS = (
    [f"ttft_p{p}" for p in PERCENTILES]
    + [f"tpot_p{p}" for p in PERCENTILES]
    + [f"e2e_p{p}" for p in PERCENTILES]
    + ["ttft_mean", "tpot_mean", "e2e_mean", "output_tok_s"]
)


@dataclass
class Measured:
    """Client-side measurements of one replay, in the same layout as a SimResult."""

    send: np.ndarray[Any, np.dtype[np.float64]]  # wall clock
    prompt_len: np.ndarray[Any, np.dtype[np.float64]]
    output_len: np.ndarray[Any, np.dtype[np.float64]]
    first_token: np.ndarray[Any, np.dtype[np.float64]]  # seconds after send
    last_token: np.ndarray[Any, np.dtype[np.float64]]
    errors: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_result(self) -> SimResult:
        """Express the measurements as a SimResult so both share one summary()."""
        t0 = float(np.min(self.send))
        arrival = self.send - t0
        nan = np.full(len(arrival), np.nan)
        return SimResult(
            arrival=arrival,
            prompt_len=self.prompt_len,
            output_len=self.output_len,
            first_scheduled=nan,
            first_token=arrival + self.first_token,
            finish=arrival + self.last_token,
            preemptions=np.zeros(len(arrival)),
            max_itl=nan,
            steps={},
        )

    def workload(self, name: str = "replay") -> Workload:
        t0 = float(np.min(self.send))
        order = np.argsort(self.send, kind="stable")
        specs = tuple(
            RequestSpec(float(self.send[i] - t0), int(self.prompt_len[i]), int(self.output_len[i]))
            for i in order
        )
        return Workload(specs, name)


def load_replay(path: str | Path) -> Measured:
    sends, plen, olen, first, last = [], [], [], [], []
    errors = 0
    meta: dict[str, Any] = {}
    with open_text(path) as f:
        for line in f:
            rec = json.loads(line)
            if "summary" in rec:
                meta = rec["summary"]
                continue
            if "ntok" in rec:  # compact record (scripts/compact_results.py)
                ntok, t_first, t_last = rec["ntok"], rec.get("first"), rec.get("last")
            else:  # raw record from bench/replay.py: every token's arrival time
                toks = rec.get("tok") or []
                ntok = len(toks)
                t_first, t_last = (toks[0], toks[-1]) if toks else (None, None)
            if rec.get("error") or ntok != rec["output_len"]:
                errors += 1
                continue
            sends.append(rec["send"])
            plen.append(rec["prompt_len"])
            olen.append(rec["output_len"])
            first.append(t_first)
            last.append(t_last)
    return Measured(
        np.array(sends),
        np.array(plen, dtype=np.float64),
        np.array(olen, dtype=np.float64),
        np.array(first),
        np.array(last),
        errors,
        meta,
    )


def parse_server_log(path: str | Path, block_size: int = 16) -> dict[str, Any]:
    with open_text(path) as f:
        text = f.read()
    out: dict[str, Any] = {}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        tokens = int(m.group(1).replace(",", ""))
        out["kv_tokens"] = tokens
        out["num_blocks"] = tokens // block_size - 1  # one block is vLLM's null block
    m = re.search(r"V1 LLM engine \(v([\w.]+)\)", text)
    if m:
        out["vllm_version"] = m.group(1)
    return out


def parse_vllm_args(args: str) -> dict[str, Any]:
    out: dict[str, Any] = dict(H200_DEFAULTS)
    toks: list[str] = []
    for t in shlex.split(args):  # accept both "--flag value" and "--flag=value"
        toks.extend(t.split("=", 1) if t.startswith("--") and "=" in t else [t])
    for i, t in enumerate(toks):
        if t == "--max-num-batched-tokens":
            out["max_num_batched_tokens"] = int(toks[i + 1])
        elif t == "--max-num-seqs":
            out["max_num_seqs"] = int(toks[i + 1])
        elif t == "--num-gpu-blocks-override":
            out["num_gpu_blocks_override"] = int(toks[i + 1])
        elif t == "--no-enable-chunked-prefill":
            out["enable_chunked_prefill"] = False
        elif t == "--long-prefill-token-threshold":
            out["long_prefill_token_threshold"] = int(toks[i + 1])
        elif t == "--max-model-len":
            out["max_model_len"] = int(toks[i + 1])
    return out


@dataclass
class RunDir:
    path: Path
    meta: dict[str, Any]
    server: dict[str, Any]
    args: dict[str, Any]
    windows: dict[str, tuple[float, float]]
    _steps: list[StepRecord] | None = None

    @classmethod
    def load(cls, path: str | Path) -> RunDir:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        windows = {}
        runs = path / "runs.jsonl"
        if runs.exists():
            for line in runs.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    windows[r["run"]] = (float(r["start"]), float(r["end"]))
        return cls(
            path,
            meta,
            parse_server_log(path / "server.log"),
            parse_vllm_args(meta.get("vllm_args", "")),
            windows,
        )

    def scheduler_config(self, max_model_len: int | None = None) -> SchedulerConfig:
        # bench/explorer.sbatch serves with --max-model-len 16384 unless overridden.
        max_model_len = max_model_len or int(self.args.get("max_model_len", 16384))
        blocks = self.args.get("num_gpu_blocks_override")
        num_blocks = blocks - 1 if blocks else self.server["num_blocks"]
        return SchedulerConfig(
            max_num_seqs=self.args["max_num_seqs"],
            max_num_batched_tokens=self.args["max_num_batched_tokens"],
            num_blocks=num_blocks,
            max_model_len=max_model_len,
            enable_chunked_prefill=self.args.get("enable_chunked_prefill", True),
            long_prefill_token_threshold=self.args.get("long_prefill_token_threshold", 0),
        )

    def steps_in(self, run: str) -> list[StepRecord]:
        if self._steps is None:
            self._steps = load_probe(self.path / "steps.jsonl")
        t0, t1 = self.windows[run]
        return [s for s in self._steps if t0 <= s.g0 <= t1]

    def runs(self, max_error_frac: float = 0.01) -> list[str]:
        """Completed replays in which at most ``max_error_frac`` of requests failed."""
        out = []
        for r in self.windows:
            path = self.path / f"{r}.jsonl"
            if not exists(path):
                continue
            m = load_replay(path)
            # A replay that lost requests (e.g. the server died mid-run) is not comparable.
            if len(m.send) and m.errors <= max_error_frac * (len(m.send) + m.errors):
                out.append(r)
        return out


def compare(
    measured: SimResult, predicted: SimResult, warmup: float, cooldown: float
) -> dict[str, dict[str, float]]:
    real = measured.summary(warmup, cooldown)
    sim = predicted.summary(warmup, cooldown)
    out: dict[str, dict[str, float]] = {}
    for k in METRICS:
        if k in real and k in sim:
            r, s = real[k], sim[k]
            out[k] = {"real": r, "sim": s, "rel_err": (s - r) / r if r else math.nan}
    return out


def validate_run(
    rd: RunDir,
    run: str,
    cost: StepTimeModel,
    frontend_latency: float = 0.0,
    idle_launch_latency: float = 0.0,
    warmup_frac: float = 0.1,
    cooldown_frac: float = 0.1,
) -> dict[str, Any]:
    meas = load_replay(rd.path / f"{run}.jsonl")
    wl = meas.workload(run)
    pred = simulate(
        wl,
        cost,
        "vllm",
        rd.scheduler_config(),
        EngineConfig(
            async_scheduling=True,
            frontend_latency=frontend_latency,
            idle_launch_latency=idle_launch_latency,
        ),
    )
    real = meas.as_result()
    span = wl.duration
    cmp = compare(real, pred, span * warmup_frac, span * cooldown_frac)
    return {
        "run": run,
        "requests": len(wl),
        "errors": meas.errors,
        "rate": wl.rate,
        "sim_preemptions": int(pred.preemptions.sum()),
        "sim_wall_s": pred.meta["wall_s"],
        "metrics": cmp,
    }


def step_model_error(
    cost: StepTimeModel,
    steps: list[StepRecord],
    y: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> dict[str, float]:
    """How well the step model predicts measured iteration periods.

    Pass ``y`` (periods computed on the full, contiguous step sequence) when ``steps`` is
    a subset, since a period depends on the step before it.
    """
    if not steps:
        return {}
    y = periods(steps) if y is None else y
    pred = np.array([cost.step_time(s.stats) for s in steps])
    ape = np.abs(pred - y) / y
    tot = float(y.sum())
    return {
        "steps": float(len(y)),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
        "time_weighted_ape": float((ape * y).sum() / tot),
        "total_time_err": float((pred.sum() - tot) / tot),
    }


def estimate_frontend(
    rd: RunDir,
    cost: StepTimeModel,
    run: str = "calibration",
    burst_window: float = 0.005,
    max_queue: float = 0.005,
) -> dict[str, float]:
    """Estimate the two latencies the engine model leaves out, from the calibration replay.

    The calibration replay is simulated with both set to zero. Requests that arrive while
    the simulated engine is busy only see the frontend path (HTTP, API server, output
    detokenization and streaming), so the median of measured minus simulated TTFT over
    them is ``frontend_latency``. Requests that find the engine idle additionally pay
    ``idle_launch_latency``: the median of what remains of their gap. Requests sent in a
    burst (another request within ``burst_window`` seconds) are left out: the API server
    admits them one at a time, a queueing delay rather than a fixed latency. So are
    requests the simulated engine did not admit within ``max_queue`` seconds: near
    saturation, a 1 % step-time error turns into tens of milliseconds of queueing, which
    is model error, not frontend latency. Validation replays are never used.
    """
    meas = load_replay(rd.path / f"{run}.jsonl")
    order = np.argsort(meas.send, kind="stable")
    wl = meas.workload(run)
    pred = simulate(wl, cost, "vllm", rd.scheduler_config(), EngineConfig(async_scheduling=True))
    gap = meas.first_token[order] - pred.ttft  # workload order is send order
    start, end = pred.steps["start"], pred.steps["start"] + pred.steps["dur"]
    k = np.searchsorted(start, pred.arrival, side="left") - 1  # steps strictly before
    busy = (k >= 0) & (pred.arrival < end[np.clip(k, 0, None)])
    gaps = np.diff(pred.arrival)
    lone = np.ones(len(gap), dtype=bool)
    lone[1:] &= gaps > burst_window
    lone[:-1] &= gaps > burst_window
    unqueued = pred.queue_delay <= max_queue
    busy_lone, idle_lone = busy & lone & unqueued, ~busy & lone & unqueued
    frontend = float(np.median(gap[busy_lone])) if busy_lone.any() else 0.0
    idle = float(np.median(gap[idle_lone]) - frontend) if idle_lone.any() else 0.0
    return {
        "frontend_latency": frontend,
        "idle_launch_latency": max(0.0, idle),
        "busy_requests": float(busy_lone.sum()),
        "idle_requests": float(idle_lone.sum()),
    }
