"""Discrete-event engine loop.

The engine alternates between asking the scheduler for the next iteration and advancing
time by the step-time model's estimate for that iteration. Time jumps straight to the next
arrival when the engine is idle, so simulated cost is proportional to the number of
iterations, not to simulated seconds.

With ``async_scheduling`` (vLLM's default since 0.11) the plan for step ``k+1`` is made
when step ``k`` starts, using request state as if ``k`` had completed (output placeholders),
but resources freed by requests finishing in ``k`` and requests that arrive during ``k`` only
become visible to the plan after that, at step ``k+2``. Without it, each step is planned
after the previous one completes.
"""

from __future__ import annotations

import math
import time
from array import array
from dataclasses import dataclass
from typing import Any

import numpy as np

from batchsim.cost import StepTimeModel
from batchsim.metrics import SimResult
from batchsim.request import Request
from batchsim.schedulers import Batch, Scheduler, SchedulerConfig, make_scheduler
from batchsim.workload import Workload

STEP_FIELDS = (
    "start",
    "dur",
    "num_tokens",
    "num_seqs",
    "num_decode",
    "num_prefill",
    "prefill_tokens",
    "kv_usage",
    "num_waiting",
    "num_running",
    "num_preempted",
)


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    async_scheduling: bool = True
    # Client-side latency the engine does not see (HTTP, API server); added to TTFT/E2E.
    frontend_latency: float = 0.0
    # Extra delay before the first iteration after the engine sat idle: the engine thread
    # wakes from its input-queue wait and does scheduling and input preparation that a
    # busy engine overlaps with the previous iteration.
    idle_launch_latency: float = 0.0
    record_itl: bool = False
    record_steps: bool = True
    max_steps: int = 50_000_000


class Simulator:
    def __init__(
        self,
        scheduler: Scheduler,
        cost_model: StepTimeModel,
        config: EngineConfig | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.cost = cost_model
        self.config = config or EngineConfig()

    def run(self, workload: Workload) -> SimResult:
        cfg = self.config
        sched = self.scheduler
        cost = self.cost.step_time
        reqs = [Request.from_spec(i, s) for i, s in enumerate(workload.specs)]
        if cfg.record_itl:
            for r in reqs:
                r.itl = []
        log = {f: array("d") for f in STEP_FIELDS} if cfg.record_steps else None
        kv = sched.kv

        def start(b: Batch, t: float) -> None:
            b.start = t
            dur = cost(b.stats)
            if not dur > 0:
                raise SimulationError(f"step-time model returned {dur!r} for {b.stats}")
            b.end = t + dur
            if log is not None:
                s = b.stats
                for name, value in (
                    ("start", t),
                    ("dur", dur),
                    ("num_tokens", s.num_tokens),
                    ("num_seqs", s.num_seqs),
                    ("num_decode", s.num_decode),
                    ("num_prefill", s.num_prefill),
                    ("prefill_tokens", s.prefill_tokens),
                    ("kv_usage", kv.usage),
                    ("num_waiting", len(sched.waiting)),
                    ("num_running", len(sched.running)),
                    ("num_preempted", b.num_preempted),
                ):
                    log[name].append(value)

        n = len(reqs)
        ai = 0
        t = reqs[0].arrival if reqs else 0.0
        idle = True
        idle_latency = cfg.idle_launch_latency
        inflight: Batch | None = None
        is_async = cfg.async_scheduling and sched.supports_async
        steps = 0
        wall0 = time.perf_counter()
        while True:
            while ai < n and reqs[ai].arrival <= t:
                sched.add_request(reqs[ai])
                ai += 1
            nxt = sched.schedule(t)
            steps += 1
            if steps > cfg.max_steps:
                raise SimulationError(f"exceeded max_steps={cfg.max_steps}")

            if inflight is None:
                if not nxt:
                    wake = min(sched.next_wakeup(t), reqs[ai].arrival if ai < n else math.inf)
                    if wake == math.inf:
                        if sched.has_work():
                            raise SimulationError(
                                f"scheduler stalled at t={t:.3f}s with "
                                f"{len(sched.waiting)} waiting / {len(sched.running)} running"
                            )
                        break
                    if wake <= t:
                        raise SimulationError(f"scheduler idle but asked to wake at {wake} <= {t}")
                    t = wake
                    idle = True
                    continue
                if idle:
                    t += idle_latency
                    idle = False
                start(nxt, t)
                if is_async:
                    inflight = nxt  # plan the following step now, while this one runs
                else:
                    t = nxt.end
                    sched.complete(nxt, t)
                continue

            # Async: `nxt` was planned at the start of `inflight`.
            t = inflight.end
            sched.complete(inflight, t)
            if nxt:
                start(nxt, t)
                inflight = nxt
            else:
                inflight = None

        wall = time.perf_counter() - wall0
        steps_np = (
            {k: np.frombuffer(v, dtype=np.float64).copy() for k, v in log.items()} if log else {}
        )
        unfinished = sum(1 for r in reqs if math.isnan(r.finish) or r.emitted < r.output_len)
        if unfinished:
            raise SimulationError(f"{unfinished} requests did not finish")
        meta: dict[str, Any] = {
            "policy": sched.name,
            "frontend_latency": cfg.frontend_latency,
            "idle_launch_latency": cfg.idle_launch_latency,
            "scheduler": sched.cfg.__dict__,
            "async_scheduling": is_async,
            "cost_model": self.cost.kind,
            "workload": workload.name,
            "wall_s": wall,
            "num_iterations": len(steps_np.get("start", ())),
            "num_preemptions": sched.num_preemptions,
        }
        return SimResult.from_requests(reqs, steps_np, cfg.frontend_latency, meta)


def simulate(
    workload: Workload,
    cost_model: StepTimeModel,
    policy: str = "vllm",
    scheduler_config: SchedulerConfig | None = None,
    engine_config: EngineConfig | None = None,
) -> SimResult:
    """Run ``workload`` through a fresh scheduler of the given policy."""
    sched = make_scheduler(policy, scheduler_config or SchedulerConfig())
    return Simulator(sched, cost_model, engine_config).run(workload)
