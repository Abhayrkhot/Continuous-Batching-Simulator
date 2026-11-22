"""Differential test: batchsim's ``vllm`` policy against vLLM's own scheduler.

Builds vLLM's real ``Scheduler`` / ``AsyncScheduler`` (CPU only; no model weights are
loaded, only the model's config) and batchsim's ``VLLMScheduler`` with the same limits,
feeds both the same randomized request streams, and compares, step by step, which requests
each schedules and how many tokens each gets. Requests arrive between steps; asynchronous
scheduling is driven the way vLLM's engine core does it (the next step is scheduled before
the previous step's output is processed, two steps in flight).

    apptainer exec vllm.sif python3 bench/scheduler_parity.py --scenarios 300 --out parity.json

Needs vLLM 0.34 and the batchsim sources on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from batchsim.request import Request as SimRequest
from batchsim.request import RequestSpec
from batchsim.schedulers import Batch, SchedulerConfig, VLLMScheduler

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 16


@dataclass(frozen=True)
class Scenario:
    seed: int
    max_num_seqs: int
    budget: int
    chunked: bool
    long_prefill: int
    num_blocks: int  # usable blocks (vLLM gets one more for its null block)
    max_model_len: int
    async_sched: bool
    requests: tuple[tuple[int, int, int], ...]  # (arrival_tick, prompt_len, output_len)


def random_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    chunked = rng.random() < 0.8
    budget = rng.choice([32, 64, 128, 256, 512])
    max_model_len = 1024 if chunked else budget
    n = rng.randint(2, 40)
    reqs = []
    for _ in range(n):
        p = rng.randint(1, min(600, max_model_len - 1))
        o = rng.randint(1, min(80, max_model_len - p))
        reqs.append((rng.randint(0, 60), p, o))
    reqs.sort(key=lambda r: r[0])
    need = max(-(-(p + o) // BLOCK) for _, p, o in reqs)
    # Tight caches force preemption; roomy ones exercise admission limits.
    num_blocks = rng.choice([need, need + 2, need * 2, need * 4, 4096])
    return Scenario(
        seed=seed,
        max_num_seqs=min(budget, rng.choice([1, 2, 4, 8, 16, 64])),  # vLLM requires <= budget
        budget=budget,
        chunked=chunked,
        long_prefill=rng.choice([0, 0, 0, 16, 64]) if chunked else 0,
        num_blocks=num_blocks,
        max_model_len=max_model_len,
        async_sched=rng.random() < 0.5,
        requests=tuple(reqs),
    )


class Adapter(Protocol):
    def add(self, idx: int) -> None: ...
    def schedule(self) -> tuple[dict[str, int], Any]: ...
    def complete(self, handle: Any) -> None: ...
    def unfinished(self) -> int: ...


class VllmAdapter:
    def __init__(self, sc: Scenario) -> None:
        import torch
        from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request
        from vllm.v1.structured_output import StructuredOutputManager

        self._mro = ModelRunnerOutput
        model_config = ModelConfig(
            model=MODEL,
            dtype="float16",
            seed=0,
            max_model_len=sc.max_model_len,
            skip_tokenizer_init=True,
        )
        sched_config = SchedulerConfig(
            max_num_seqs=sc.max_num_seqs,
            max_num_batched_tokens=sc.budget,
            max_model_len=sc.max_model_len,
            long_prefill_token_threshold=sc.long_prefill,
            enable_chunked_prefill=sc.chunked,
            async_scheduling=sc.async_sched,
            is_encoder_decoder=False,
            watermark=0.0,
        )
        cache_config = CacheConfig(
            block_size=BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        )
        vllm_config = VllmConfig(
            scheduler_config=sched_config,
            model_config=model_config,
            cache_config=cache_config,
            device_config=DeviceConfig(device="cpu"),  # scheduling needs no GPU
        )
        spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
        nb = sc.num_blocks + 1  # block 0 is vLLM's null block
        kv_config = KVCacheConfig(
            num_blocks=nb, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)]
        )
        cache_config.num_gpu_blocks = nb
        register_all_kvcache_specs(vllm_config)
        cls = AsyncScheduler if sc.async_sched else Scheduler
        self.s = cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_config,
            block_size=BLOCK,
            log_stats=False,
            structured_output_manager=StructuredOutputManager(vllm_config),
        )
        rng = random.Random(sc.seed)
        self.reqs = []
        for i, (_, p, o) in enumerate(sc.requests):
            sp = SamplingParams(ignore_eos=True, max_tokens=o, temperature=0.0)
            sp.update_from_generation_config({}, 151643)
            ids = [rng.randrange(1000, 100000) for _ in range(p)]
            self.reqs.append(Request(str(i), ids, sp, None, arrival_time=float(i)))

    def add(self, idx: int) -> None:
        self.s.add_request(self.reqs[idx])

    def schedule(self) -> tuple[dict[str, int], Any]:
        so = self.s.schedule()
        sched = dict(so.num_scheduled_tokens)
        samples = {rid for rid in sched if not self.s.requests[rid].is_prefill_chunk}
        return sched, (so, samples)

    def complete(self, handle: Any) -> None:
        so, samples = handle
        rids = list(so.num_scheduled_tokens)
        out = self._mro(req_ids=rids, sampled_token_ids=[[7] if r in samples else [] for r in rids])
        self.s.update_from_output(so, out)

    def unfinished(self) -> int:
        return int(self.s.get_num_unfinished_requests())


class SimAdapter:
    def __init__(self, sc: Scenario) -> None:
        cfg = SchedulerConfig(
            max_num_seqs=sc.max_num_seqs,
            max_num_batched_tokens=sc.budget,
            num_blocks=sc.num_blocks,
            block_size=BLOCK,
            max_model_len=sc.max_model_len,
            enable_chunked_prefill=sc.chunked,
            long_prefill_token_threshold=sc.long_prefill,
        )
        self.s = VLLMScheduler(cfg)
        self.reqs = [
            SimRequest.from_spec(i, RequestSpec(0.0, p, o))
            for i, (_, p, o) in enumerate(sc.requests)
        ]
        self.tick = 0.0

    def add(self, idx: int) -> None:
        self.s.add_request(self.reqs[idx])

    def schedule(self) -> tuple[dict[str, int], Any]:
        b = self.s.schedule(self.tick)
        return {str(r.rid): n for r, n in zip(b.reqs, b.tokens)}, b

    def complete(self, handle: Batch) -> None:
        self.tick += 1.0
        self.s.complete(handle, self.tick)

    def unfinished(self) -> int:
        return sum(1 for r in self.reqs if r.state != 2)


def drive(a: Adapter, sc: Scenario, max_ticks: int = 20000) -> list[dict[str, int]]:
    """Run to completion; return the per-step schedules (empty steps omitted)."""
    by_tick: dict[int, list[int]] = {}
    for i, (t, _, _) in enumerate(sc.requests):
        by_tick.setdefault(t, []).append(i)
    last_arrival = max(by_tick)
    trace: list[dict[str, int]] = []
    inflight: deque[Any] = deque()
    for tick in range(max_ticks):
        for i in by_tick.get(tick, ()):
            a.add(i)
        sched, handle = a.schedule()
        if sched:
            trace.append(sched)
        if not sc.async_sched:
            if sched:
                a.complete(handle)
        else:
            if sched:
                inflight.append(handle)
            if len(inflight) >= 2 or (inflight and not sched):
                a.complete(inflight.popleft())
        if tick > last_arrival and not inflight and a.unfinished() == 0:
            return trace
    raise RuntimeError(f"scenario {sc.seed} did not finish in {max_ticks} ticks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()
    results = []
    totals = {"scenarios": 0, "steps": 0, "mismatched_scenarios": 0, "preempting_scenarios": 0}
    for k in range(args.scenarios):
        sc = random_scenario(args.seed + k)
        sim = SimAdapter(sc)
        ref = drive(VllmAdapter(sc), sc)
        got = drive(sim, sc)
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
            None if len(ref) == len(got) else min(len(ref), len(got)),
        )
        preempt = sim.s.num_preemptions
        totals["scenarios"] += 1
        totals["steps"] += len(ref)
        totals["mismatched_scenarios"] += first_diff is not None
        totals["preempting_scenarios"] += preempt > 0
        row = {
            "seed": sc.seed,
            "async": sc.async_sched,
            "chunked": sc.chunked,
            "steps": len(ref),
            "preemptions": preempt,
            "match": first_diff is None,
        }
        if first_diff is not None:
            row["first_diff_step"] = first_diff
            row["vllm"] = ref[first_diff] if first_diff < len(ref) else None
            row["batchsim"] = got[first_diff] if first_diff < len(got) else None
            print(f"MISMATCH seed={sc.seed}: {row}", file=sys.stderr)
        results.append(row)
    totals["wall_s"] = round(time.time() - t0, 1)
    import vllm

    totals["vllm_version"] = vllm.__version__
    print(json.dumps(totals))
    if args.out:
        args.out.write_text(json.dumps({"totals": totals, "scenarios": results}, indent=1) + "\n")
    sys.exit(1 if totals["mismatched_scenarios"] else 0)


if __name__ == "__main__":
    main()
