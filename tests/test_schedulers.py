"""Scheduler decisions on small, hand-checked cases."""

from __future__ import annotations

import pytest

from batchsim.request import Request, RequestSpec
from batchsim.schedulers import (
    Batch,
    OrcaScheduler,
    PrefillFirstScheduler,
    Scheduler,
    SchedulerConfig,
    StaticScheduler,
    VLLMScheduler,
    make_scheduler,
)


def req(rid: int, prompt: int, output: int, arrival: float = 0.0) -> Request:
    return Request.from_spec(rid, RequestSpec(arrival, prompt, output))


def plan(b: Batch) -> dict[int, int]:
    return {r.rid: n for r, n in zip(b.reqs, b.tokens)}


def run_steps(s, max_steps: int = 1000) -> list[dict[int, int]]:
    """Drive a scheduler synchronously until idle; return the per-step plans."""
    out = []
    t = 0.0
    for _ in range(max_steps):
        b = s.schedule(t)
        if not b:
            break
        t += 1.0
        s.complete(b, t)
        out.append(plan(b))
    return out


class TestVLLM:
    def test_chunked_prefill_then_decode(self) -> None:
        s = VLLMScheduler(SchedulerConfig(max_num_batched_tokens=4, num_blocks=100, block_size=4))
        r = req(0, 10, 3)
        s.add_request(r)
        steps = run_steps(s)
        assert steps == [{0: 4}, {0: 4}, {0: 2}, {0: 1}, {0: 1}]
        assert r.first_token == 3.0  # the third (last) prefill chunk samples token 1
        assert r.finish == 5.0
        assert r.emitted == 3

    def test_running_first_then_fcfs_admission_with_shared_budget(self) -> None:
        s = VLLMScheduler(SchedulerConfig(max_num_batched_tokens=8, num_blocks=100, block_size=4))
        a, b = req(0, 6, 3), req(1, 6, 2)
        s.add_request(a)
        s.add_request(b)
        steps = run_steps(s)
        # Step 1: a's whole prompt (6) + 2 tokens of b. Step 2: a decodes, b finishes its
        # prompt. Step 3: both decode their last token.
        assert steps == [{0: 6, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 1}]
        assert a.emitted == 3 and b.emitted == 2

    def test_unchunked_waits_for_budget(self) -> None:
        cfg = SchedulerConfig(
            max_num_batched_tokens=8,
            num_blocks=100,
            block_size=4,
            enable_chunked_prefill=False,
            max_model_len=8,
        )
        s = VLLMScheduler(cfg)
        s.add_request(req(0, 6, 2))
        s.add_request(req(1, 6, 2))
        steps = run_steps(s)
        assert steps[0] == {0: 6}  # 6 + 6 > 8 and no chunking: b waits
        assert steps[1] == {0: 1, 1: 6}

    def test_max_num_seqs(self) -> None:
        s = VLLMScheduler(SchedulerConfig(max_num_seqs=2, num_blocks=100))
        for i in range(4):
            s.add_request(req(i, 4, 3))
        for p in run_steps(s):
            assert len(p) <= 2

    def test_long_prefill_threshold_caps_chunks(self) -> None:
        cfg = SchedulerConfig(
            max_num_batched_tokens=64, num_blocks=100, long_prefill_token_threshold=8
        )
        s = VLLMScheduler(cfg)
        s.add_request(req(0, 20, 1))
        s.add_request(req(1, 20, 1))
        assert run_steps(s)[0] == {0: 8, 1: 8}

    def test_reserve_full_isl_blocks_admission(self) -> None:
        # 4 blocks of 4 tokens. a takes one; b's 13-token prompt needs 4 more, so it is not
        # admitted, although its first 4-token chunk would fit.
        s_no = VLLMScheduler(
            SchedulerConfig(
                max_num_batched_tokens=8, num_blocks=4, block_size=4, reserve_full_isl=False
            )
        )
        s_no.add_request(req(0, 4, 1))
        s_no.add_request(req(1, 13, 1))
        assert plan(s_no.schedule(0.0)) == {0: 4, 1: 4}
        s_yes = VLLMScheduler(SchedulerConfig(max_num_batched_tokens=8, num_blocks=4, block_size=4))
        s_yes.add_request(req(0, 4, 1))
        s_yes.add_request(req(1, 13, 1))
        assert plan(s_yes.schedule(0.0)) == {0: 4}

    def test_lifo_preemption_and_recompute(self) -> None:
        # 4 blocks of 4 tokens. Two requests with 7-token prompts grow into each other.
        s = VLLMScheduler(SchedulerConfig(max_num_batched_tokens=64, num_blocks=4, block_size=4))
        a, b = req(0, 7, 6), req(1, 7, 6)
        s.add_request(a)
        s.add_request(b)
        steps = run_steps(s)
        assert steps[0] == {0: 7, 1: 7}
        assert b.num_preemptions >= 1 and a.num_preemptions == 0  # newest is preempted
        assert a.emitted == 6 and b.emitted == 6
        # b's recompute re-prefills its prompt plus the tokens it had generated.
        recompute = next(p[1] for p in steps[1:] if p.get(1, 0) > 1)
        assert recompute > 7
        assert s.kv.free == s.kv.num_blocks

    def test_no_admission_in_a_preempting_step(self) -> None:
        s = VLLMScheduler(SchedulerConfig(max_num_batched_tokens=64, num_blocks=4, block_size=4))
        s.add_request(req(0, 7, 6))
        s.add_request(req(1, 7, 6))
        t = 0.0
        while True:
            b = s.schedule(t)
            if b.num_preempted:
                assert all(r.rid == 0 for r in b.reqs)
                break
            t += 1
            s.complete(b, t)

    def test_unchunked_budget_must_cover_max_model_len(self) -> None:
        with pytest.raises(ValueError, match="chunked prefill is disabled"):
            VLLMScheduler(SchedulerConfig(max_num_batched_tokens=64, enable_chunked_prefill=False))

    def test_rejects_requests_that_can_never_run(self) -> None:
        s = VLLMScheduler(SchedulerConfig(num_blocks=2, block_size=4, max_model_len=100))
        with pytest.raises(ValueError, match="KV"):
            s.add_request(req(0, 9, 1))
        with pytest.raises(ValueError, match="max_model_len"):
            s.add_request(req(1, 90, 20))


class TestBaselines:
    def test_prefill_first_never_mixes(self) -> None:
        s = PrefillFirstScheduler(
            SchedulerConfig(max_num_batched_tokens=64, max_model_len=64, num_blocks=100)
        )
        s.add_request(req(0, 5, 4))
        t = 0.0
        b = s.schedule(t)
        s.complete(b, 1.0)
        s.add_request(req(1, 5, 4))
        b2 = s.schedule(1.0)
        assert plan(b2) == {1: 5}  # prefill-only step; request 0's decode waits
        s.complete(b2, 2.0)
        assert plan(s.schedule(2.0)) == {0: 1, 1: 1}

    def test_prefill_first_requires_budget_for_max_len(self) -> None:
        with pytest.raises(ValueError):
            PrefillFirstScheduler(SchedulerConfig(max_num_batched_tokens=10, max_model_len=100))

    def test_orca_reservation_caps_concurrency(self) -> None:
        # Each request reserves prompt+output = 16 tokens = 4 blocks; 8 blocks fit two.
        s = OrcaScheduler(
            SchedulerConfig(max_num_batched_tokens=1 << 30, num_blocks=8, block_size=4)
        )
        for i in range(3):
            s.add_request(req(i, 8, 8))
        steps = run_steps(s)
        assert steps[0] == {0: 8, 1: 8}
        assert max(len(p) for p in steps) == 2
        assert s.kv.free == 8

    def test_orca_reserve_max_tokens(self) -> None:
        cfg = SchedulerConfig(
            max_num_batched_tokens=1 << 30, num_blocks=8, block_size=4, reserve_max_tokens=24
        )
        s = OrcaScheduler(cfg)
        s.add_request(req(0, 8, 2))
        s.add_request(req(1, 8, 2))
        assert plan(s.schedule(0.0)) == {0: 8}  # 8 + 24 tokens = all 8 blocks

    def test_static_pads_and_releases_together(self) -> None:
        s = StaticScheduler(SchedulerConfig(max_num_seqs=2, num_blocks=100, block_size=4))
        a, b = req(0, 3, 2), req(1, 7, 5)
        s.add_request(a)
        s.add_request(b)
        first = s.schedule(0.0)
        assert first.stats.num_tokens == 14  # both padded to 7
        s.complete(first, 1.0)
        t = 1.0
        while s.running:
            bt = s.schedule(t)
            t += 1.0
            s.complete(bt, t)
        assert a.finish == b.finish == 5.0  # 1 prefill + 4 decode steps for the longest
        assert a.emitted == 2 and b.emitted == 5

    def test_static_waits_for_full_batch_or_timeout(self) -> None:
        s = StaticScheduler(SchedulerConfig(max_num_seqs=4, num_blocks=100, max_wait=0.5))
        s.add_request(req(0, 4, 2, arrival=0.0))
        assert not s.schedule(0.1)
        assert s.next_wakeup(0.1) == pytest.approx(0.5)
        assert s.schedule(0.5)


def test_registry() -> None:
    assert set(Scheduler.registry) == {
        "vllm",
        "prefill-first",
        "orca",
        "static",
    }
    with pytest.raises(ValueError):
        make_scheduler("nope", SchedulerConfig())


def test_reserve_never_below_actual_use() -> None:
    # reserve_max_tokens (8) is below the true output (400): reserve what will be used.
    cfg = SchedulerConfig(
        max_num_batched_tokens=1 << 30, num_blocks=60, block_size=16, reserve_max_tokens=8
    )
    s = OrcaScheduler(cfg)
    for i in range(4):
        s.add_request(req(i, 16, 400))
    steps = run_steps(s, max_steps=5000)
    assert max(len(p) for p in steps) == 2  # 26 blocks each; two fit in 60
    assert s.kv.peak_used <= 60


def test_declared_max_tokens_wins() -> None:
    cfg = SchedulerConfig(
        max_num_batched_tokens=1 << 30, num_blocks=100, block_size=4, reserve_max_tokens=8
    )
    s = OrcaScheduler(cfg)
    r = Request.from_spec(0, RequestSpec(0.0, 8, 2, max_tokens=100))
    s.add_request(r)
    s.schedule(0.0)
    assert r.num_blocks == -(-(8 + 100) // 4)


def test_prefill_first_watermark_does_not_block_an_idle_engine() -> None:
    cfg = SchedulerConfig(
        max_num_batched_tokens=256, max_model_len=256, num_blocks=10, block_size=16, watermark=0.2
    )
    s = PrefillFirstScheduler(cfg)
    r = req(0, 150, 5)
    s.add_request(r)
    run_steps(s)
    assert r.emitted == 5
