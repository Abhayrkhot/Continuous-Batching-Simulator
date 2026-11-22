"""Invariants every policy must keep, checked on random workloads (Hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from batchsim.cost import LinearStepModel
from batchsim.engine import EngineConfig, Simulator
from batchsim.request import RequestSpec
from batchsim.schedulers import SchedulerConfig, make_scheduler
from batchsim.workload import Workload

COST = LinearStepModel.simple(const=0.01, per_token=1e-4, decode_ctx_m=0.5)


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[str, SchedulerConfig, Workload, bool]:
    policy = draw(st.sampled_from(["vllm", "prefill-first", "orca", "static"]))
    n = draw(st.integers(1, 25))
    reqs = sorted(
        (
            draw(st.floats(0, 3, allow_nan=False)),
            draw(st.integers(1, 120)),
            draw(st.integers(1, 40)),
        )
        for _ in range(n)
    )
    block = draw(st.sampled_from([1, 4, 16]))
    biggest = max(-(-(p + o) // block) for _, p, o in reqs)
    num_blocks = draw(st.sampled_from([biggest, biggest + 3, biggest * 3, 10_000]))
    budget = draw(st.sampled_from([8, 32, 128, 1024]))
    if policy == "prefill-first":
        budget = max(budget, 256)
    if policy == "static":
        # a padded batch must fit: give every slot the biggest footprint
        num_blocks = max(num_blocks, biggest)
    cfg = SchedulerConfig(
        max_num_seqs=draw(st.sampled_from([1, 2, 5, 64])),
        max_num_batched_tokens=budget,
        num_blocks=num_blocks,
        block_size=block,
        max_model_len=256,
        enable_chunked_prefill=draw(st.booleans()) or policy != "vllm",
        max_wait=draw(st.sampled_from([0.0, 0.2])),
    )
    if policy == "vllm" and not cfg.enable_chunked_prefill:
        cfg = cfg.with_(max_num_batched_tokens=max(budget, cfg.max_model_len))  # as vLLM requires
    w = Workload(tuple(RequestSpec(a, p, o) for a, p, o in reqs), "prop")
    return policy, cfg, w, draw(st.booleans())


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_invariants(sc: tuple[str, SchedulerConfig, Workload, bool]) -> None:
    policy, cfg, w, is_async = sc
    sched = make_scheduler(policy, cfg)
    res = Simulator(sched, COST, EngineConfig(async_scheduling=is_async)).run(w)

    # Every request finishes, after it arrives, with all its tokens.
    assert np.all(res.first_token > res.arrival)
    assert np.all(res.finish >= res.first_token)
    # KV accounting is exact and never oversubscribed.
    assert sched.kv.free == cfg.num_blocks
    assert sched.kv.peak_used <= cfg.num_blocks
    assert not sched.waiting and not sched.running

    st_ = res.steps
    assert np.all(st_["num_seqs"] <= cfg.max_num_seqs)
    assert np.all(st_["dur"] > 0)
    if policy == "vllm":
        assert np.all(st_["num_tokens"] <= cfg.max_num_batched_tokens)

    # Token conservation: each request computes its prompt and all but its last output
    # token once, plus whatever recompute preemption forced.
    need = float((res.prompt_len + res.output_len - 1).sum())
    computed = float(st_["num_tokens"].sum())
    if policy == "static":
        assert computed >= need
    elif res.meta["num_preemptions"] == 0:
        assert computed == need
    else:
        assert computed > need
