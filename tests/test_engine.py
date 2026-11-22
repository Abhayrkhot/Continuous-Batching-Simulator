"""Engine timing: idle jumps, asynchronous planning, static batching timers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from batchsim.cost import BatchStats, LinearStepModel, StepTimeModel
from batchsim.engine import EngineConfig, SimulationError, simulate
from batchsim.schedulers import SchedulerConfig
from tests.conftest import wl

SYNC = EngineConfig(async_scheduling=False)
ASYNC = EngineConfig(async_scheduling=True)


@pytest.mark.parametrize("eng", [SYNC, ASYNC])
def test_single_request_timing(unit_cost: LinearStepModel, eng: EngineConfig) -> None:
    res = simulate(wl((0.0, 5, 4)), unit_cost, "vllm", SchedulerConfig(num_blocks=100), eng)
    assert res.ttft[0] == 1.0  # one prefill iteration
    assert res.e2e[0] == 4.0  # then three decodes
    assert res.tpot[0] == 1.0
    assert res.meta["num_iterations"] == 4


def test_async_planning_delays_arrivals_by_one_step(unit_cost: LinearStepModel) -> None:
    # B arrives mid-way through A's first iteration [0, 1).
    w = wl((0.0, 1, 10), (0.5, 1, 1))
    cfg = SchedulerConfig(num_blocks=100)
    sync = simulate(w, unit_cost, "vllm", cfg, SYNC)
    asyn = simulate(w, unit_cost, "vllm", cfg, ASYNC)
    assert sync.first_token[1] == 2.0  # planned at t=1, after B arrived
    assert asyn.first_token[1] == 3.0  # step 2 was planned at t=0; B joins step 3


def test_async_equals_sync_without_contention(token_cost: LinearStepModel) -> None:
    w = wl(*[(0.0, 10 + 7 * i, 1 + 3 * i) for i in range(12)])
    cfg = SchedulerConfig(num_blocks=10_000, max_num_batched_tokens=64)
    a = simulate(w, token_cost, "vllm", cfg, ASYNC)
    s = simulate(w, token_cost, "vllm", cfg, SYNC)
    np.testing.assert_allclose(a.first_token, s.first_token)
    np.testing.assert_allclose(a.finish, s.finish)


def test_idle_gap_is_skipped(unit_cost: LinearStepModel) -> None:
    res = simulate(
        wl((0.0, 3, 2), (100.0, 3, 2)), unit_cost, "vllm", SchedulerConfig(num_blocks=100)
    )
    assert res.finish.tolist() == [2.0, 102.0]
    assert res.meta["num_iterations"] == 4
    assert res.summary()["busy_frac"] == pytest.approx(4 / 102)


def test_static_timeout_launches_partial_batch(unit_cost: LinearStepModel) -> None:
    cfg = SchedulerConfig(max_num_seqs=4, num_blocks=100, max_wait=0.5)
    res = simulate(wl((0.0, 4, 2)), unit_cost, "static", cfg)
    assert res.first_token[0] == 1.5
    assert res.finish[0] == 2.5


def test_deterministic(token_cost: LinearStepModel) -> None:
    w = wl(*[(0.1 * i, 50 + i, 5 + i % 7) for i in range(40)])
    cfg = SchedulerConfig(num_blocks=40, block_size=16, max_num_batched_tokens=128)
    a = simulate(w, token_cost, "vllm", cfg)
    b = simulate(w, token_cost, "vllm", cfg)
    np.testing.assert_array_equal(a.finish, b.finish)
    assert a.meta["num_preemptions"] == b.meta["num_preemptions"]


def test_step_log(token_cost: LinearStepModel) -> None:
    res = simulate(
        wl((0.0, 20, 3), (0.0, 30, 2)), token_cost, "vllm", SchedulerConfig(num_blocks=100)
    )
    st = res.steps
    assert st["num_tokens"].tolist() == [50.0, 2.0, 1.0]
    assert st["num_prefill"].tolist() == [2.0, 0.0, 0.0]
    np.testing.assert_allclose(st["dur"], 0.01 + 1e-4 * st["num_tokens"])
    assert np.all(np.diff(st["start"]) > 0)


class _Broken(StepTimeModel):
    kind = ""

    def step_time(self, s: BatchStats) -> float:
        return 0.0

    def to_dict(self) -> dict[str, object]:
        return {}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> _Broken:
        return cls()


def test_rejects_non_positive_step_time() -> None:
    with pytest.raises(SimulationError, match="step-time"):
        simulate(wl((0.0, 3, 2)), _Broken(), "vllm", SchedulerConfig(num_blocks=10))


def test_empty_workload(unit_cost: LinearStepModel) -> None:
    from batchsim.workload import Workload

    res = simulate(Workload(()), unit_cost)
    assert len(res) == 0
    assert res.summary() == {"requests": 0.0}


def test_record_itl(token_cost: LinearStepModel) -> None:
    res = simulate(
        wl((0.0, 10, 5)),
        token_cost,
        "vllm",
        SchedulerConfig(num_blocks=10),
        EngineConfig(record_itl=True),
    )
    assert res.itl is not None and len(res.itl[0]) == 4
    assert math.isclose(sum(res.itl[0]), res.finish[0] - res.first_token[0])


def test_idle_launch_latency_applies_only_after_idle(unit_cost: LinearStepModel) -> None:
    eng = EngineConfig(idle_launch_latency=0.25)
    w = wl((0.0, 3, 3), (0.5, 3, 1), (100.0, 3, 1))
    res = simulate(w, unit_cost, "vllm", SchedulerConfig(num_blocks=100), eng)
    assert res.first_token[0] == 1.25  # first launch pays it
    assert res.first_token[2] == 101.25  # so does the launch after the long idle gap
    # request 1 arrives while the engine is busy: no extra delay beyond async planning
    base = simulate(w, unit_cost, "vllm", SchedulerConfig(num_blocks=100))
    assert res.first_token[1] - res.first_token[0] == base.first_token[1] - base.first_token[0]
