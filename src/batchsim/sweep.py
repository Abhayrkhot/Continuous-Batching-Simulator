"""Rate sweeps and capacity search (the largest request rate that still meets an SLO)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from batchsim.cost import StepTimeModel
from batchsim.engine import EngineConfig, simulate
from batchsim.metrics import SLO
from batchsim.schedulers import SchedulerConfig
from batchsim.workload import Workload


@dataclass(frozen=True)
class Scenario:
    """Everything needed to simulate one point; picklable for process pools."""

    workload: Workload
    cost: StepTimeModel
    policy: str = "vllm"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    warmup_frac: float = 0.1
    cooldown_frac: float = 0.1

    def run(self, slo: SLO | None = None) -> dict[str, float]:
        res = simulate(self.workload, self.cost, self.policy, self.scheduler, self.engine)
        span = self.workload.duration
        out = res.summary(span * self.warmup_frac, span * self.cooldown_frac, slo)
        out["rate"] = self.workload.rate
        return out


def _run(args: tuple[Scenario, SLO | None]) -> dict[str, float]:
    return args[0].run(args[1])


def run_many(
    scenarios: Sequence[Scenario], slo: SLO | None = None, workers: int = 1
) -> list[dict[str, float]]:
    if workers <= 1 or len(scenarios) <= 1:
        return [s.run(slo) for s in scenarios]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, [(s, slo) for s in scenarios]))


def rate_sweep(
    base: Workload,
    rates: Sequence[float],
    cost: StepTimeModel,
    policy: str = "vllm",
    scheduler: SchedulerConfig | None = None,
    engine: EngineConfig | None = None,
    slo: SLO | None = None,
    workers: int = 1,
) -> list[dict[str, float]]:
    scen = [
        Scenario(
            base.scale_to_rate(r),
            cost,
            policy,
            scheduler or SchedulerConfig(),
            engine or EngineConfig(record_steps=True),
        )
        for r in rates
    ]
    return run_many(scen, slo, workers)


def max_rate(
    meets: Callable[[float], bool],
    lo: float,
    hi: float,
    rel_tol: float = 0.02,
    max_iter: int = 30,
) -> float:
    """Largest rate in ``[lo, hi]`` with ``meets(rate)``, by bisection (assumes monotone).

    Returns 0 if even ``lo`` fails and ``hi`` if ``hi`` passes.
    """
    if not meets(lo):
        return 0.0
    if meets(hi):
        return hi
    for _ in range(max_iter):
        if hi - lo <= rel_tol * lo:
            break
        mid = math.sqrt(lo * hi)
        if meets(mid):
            lo = mid
        else:
            hi = mid
    return lo


def capacity(
    base: Workload,
    cost: StepTimeModel,
    slo: SLO,
    attainment: float = 0.9,
    policy: str = "vllm",
    scheduler: SchedulerConfig | None = None,
    engine: EngineConfig | None = None,
    lo: float = 0.1,
    hi: float = 200.0,
    rel_tol: float = 0.02,
) -> dict[str, Any]:
    """Goodput: the highest rate at which ``attainment`` of requests meet ``slo``."""
    sched = scheduler or SchedulerConfig()
    eng = engine or EngineConfig(record_steps=False)
    cache: dict[float, dict[str, float]] = {}

    def meets(rate: float) -> bool:
        s = Scenario(base.scale_to_rate(rate), cost, policy, sched, eng).run(slo)
        cache[rate] = s
        return s.get("slo_attainment", 0.0) >= attainment

    best = max_rate(meets, lo, hi, rel_tol)
    return {"rate": best, "summary": cache.get(best, {}), "evaluated": len(cache)}
