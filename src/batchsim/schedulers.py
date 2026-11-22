"""Batching policies.

Each scheduler owns a waiting queue, a running list and a :class:`BlockPool`, and turns the
current state into the next iteration's :class:`Batch`. Token accounting is applied when a
step is *scheduled* (like vLLM's ``_update_after_schedule`` with output placeholders), and
tokens are *emitted* when the step completes, which is what lets the engine model vLLM's
asynchronous scheduling: step ``k+1`` is planned while step ``k`` is still running.

Policies, from oldest to newest:

``static``
    Request-level batching (FasterTransformer, early TGI, ``generate()`` loops): a batch of
    up to ``max_num_seqs`` requests is formed, padded to its longest prompt and run until its
    longest output finishes. Nobody joins or leaves mid-batch.
``orca``
    Iteration-level scheduling (Yu et al., OSDI 2022): requests join and leave at every
    iteration and whole prompts are batched with decodes, but KV memory is reserved up front
    for ``prompt + max_tokens``, which caps concurrency.
``prefill-first``
    Paged KV with on-demand blocks and recompute preemption, and prefill-only iterations
    that take priority over decodes (vLLM's V0 default scheduler).
``vllm``
    vLLM V1 (0.34): one token budget per iteration shared by running requests (decodes and
    prefill chunks, in running order) and then FCFS admissions, chunked prefill, LIFO
    recompute preemption, and the full-prompt KV fit check on admission. With
    ``enable_chunked_prefill=False`` it is V1's unchunked mode.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, replace
from typing import ClassVar

from batchsim.cost import BatchStats
from batchsim.kvcache import BlockPool
from batchsim.request import Request

WAITING, RUNNING, FINISHED = 0, 1, 2


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    num_blocks: int = 8192
    block_size: int = 16
    max_model_len: int = 32768
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    # Fraction of blocks kept free when admitting while other requests run.
    watermark: float = 0.0
    # vLLM's scheduler_reserve_full_isl: admit only if the whole prompt fits.
    reserve_full_isl: bool = True
    # Reservation policies provision prompt + max_tokens; requests without a declared
    # max_tokens use this (None: their true output length, an oracle).
    reserve_max_tokens: int | None = None
    # Static batching: launch a partial batch once its oldest request waited this long.
    max_wait: float = 0.0

    def with_(self, **kw: object) -> SchedulerConfig:
        return replace(self, **kw)  # type: ignore[arg-type]


class Batch:
    """One iteration: which requests run how many tokens, and who samples a token."""

    __slots__ = ("end", "num_preempted", "reqs", "sampled", "start", "stats", "tokens")

    def __init__(self) -> None:
        self.reqs: list[Request] = []
        self.tokens: list[int] = []
        self.sampled: list[Request] = []
        self.stats = BatchStats()
        self.num_preempted = 0
        self.start = math.nan
        self.end = math.nan

    def __len__(self) -> int:
        return len(self.reqs)

    def __bool__(self) -> bool:
        return bool(self.reqs)


class Scheduler(ABC):
    name: ClassVar[str]
    # Whether the policy can plan step k+1 before step k completes (vLLM V1 only).
    supports_async: ClassVar[bool] = False
    registry: ClassVar[dict[str, type[Scheduler]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            Scheduler.registry[cls.name] = cls

    def __init__(self, config: SchedulerConfig) -> None:
        self.cfg = config
        self.kv = BlockPool(config.num_blocks, config.block_size)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_preemptions = 0
        self._finished_in_step = False

    # ------------------------------------------------------------------ interface

    def add_request(self, req: Request) -> None:
        total = req.prompt_len + req.output_len
        if total > self.cfg.max_model_len:
            raise ValueError(
                f"request {req.rid}: prompt+output={total} exceeds max_model_len "
                f"{self.cfg.max_model_len}; clip the workload first"
            )
        if self.kv.blocks_for(self._max_footprint(req)) > self.kv.num_blocks:
            raise ValueError(
                f"request {req.rid} needs {self.kv.blocks_for(self._max_footprint(req))} KV "
                f"blocks but the cache has {self.kv.num_blocks}"
            )
        req.state = WAITING
        self.waiting.append(req)

    @abstractmethod
    def schedule(self, now: float) -> Batch:
        """Plan the next iteration at time ``now`` (may be empty)."""

    def next_wakeup(self, now: float) -> float:
        """Earliest time an empty schedule could change without a new arrival."""
        return math.inf

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def complete(self, batch: Batch, t: float) -> list[Request]:
        """Emit the tokens produced by ``batch`` at time ``t``; return finished requests."""
        finished: list[Request] = []
        for r in batch.sampled:
            r.emitted += 1
            if r.emitted == 1:
                r.first_token = t
            else:
                gap = t - r.last_token
                if gap > r.max_itl:
                    r.max_itl = gap
                if r.itl is not None:
                    r.itl.append(gap)
            r.last_token = t
            if r.emitted == r.output_len:
                finished.append(r)
        if finished:
            self._finish(finished, t)
        return finished

    # ------------------------------------------------------------------ helpers

    def _max_footprint(self, req: Request) -> int:
        return req.prompt_len + req.output_len

    def _reserve(self, req: Request) -> int:
        """Tokens a reservation policy provisions for ``req`` at admission.

        A declared ``max_tokens`` wins; otherwise ``reserve_max_tokens`` (capped by
        ``max_model_len``); never less than the request will actually use.
        """
        need = req.prompt_len + req.output_len
        if req.declared_max_tokens or self.cfg.reserve_max_tokens is None:
            return max(need, req.reserve_tokens)
        cap = min(req.prompt_len + self.cfg.reserve_max_tokens, self.cfg.max_model_len)
        return max(need, cap)

    def _finish(self, finished: list[Request], t: float) -> None:
        drop_waiting = False
        for r in finished:
            r.finish = t
            self.kv.release(r)
            if r.state == WAITING:  # preempted while its last token was in flight
                drop_waiting = True
            r.state = FINISHED
        self.running = [r for r in self.running if r.state == RUNNING]
        if drop_waiting:
            self.waiting = deque(r for r in self.waiting if r.state == WAITING)

    def _commit(self, batch: Batch, r: Request, n: int, now: float) -> None:
        """Add ``n`` tokens of ``r`` to ``batch`` and advance its accounting."""
        computed = r.num_computed
        batch.reqs.append(r)
        batch.tokens.append(n)
        batch.stats.add(n, computed)
        if r.first_scheduled != r.first_scheduled:  # NaN check
            r.first_scheduled = now
        r.num_computed = computed + n
        if r.num_computed == r.prompt_len + r.num_generated:
            r.num_generated += 1
            batch.sampled.append(r)
            if r.num_generated == r.output_len:
                r.done = True

    def _preempt(self, r: Request) -> None:
        self.kv.release(r)
        r.num_computed = 0
        r.num_preemptions += 1
        r.state = WAITING
        self.num_preemptions += 1
        self.waiting.appendleft(r)


class VLLMScheduler(Scheduler):
    """vLLM V1 ``Scheduler.schedule()`` (vLLM 0.34, FCFS, no prefix caching/spec decode)."""

    name = "vllm"
    supports_async = True

    def __init__(self, config: SchedulerConfig) -> None:
        # vLLM's SchedulerConfig.verify_max_model_len: without chunking, a prompt (or a
        # preempted request's recompute) longer than the budget could never be scheduled.
        if (
            not config.enable_chunked_prefill
            and config.max_num_batched_tokens < config.max_model_len
        ):
            raise ValueError(
                f"max_num_batched_tokens ({config.max_num_batched_tokens}) must be >= "
                f"max_model_len ({config.max_model_len}) when chunked prefill is disabled"
            )
        super().__init__(config)

    def schedule(self, now: float) -> Batch:
        cfg, kv = self.cfg, self.kv
        batch = Batch()
        budget = cfg.max_num_batched_tokens
        running = self.running
        waiting = self.waiting
        lpt = cfg.long_prefill_token_threshold
        if not cfg.enable_chunked_prefill or len(running) + len(waiting) <= 1:
            lpt = 0
        preempted = False

        # Running requests first, in admission order.
        i = 0
        while i < len(running) and budget > 0:
            r = running[i]
            if r.done:  # its last token is in flight
                i += 1
                continue
            n = r.prompt_len + r.num_generated - r.num_computed
            if 0 < lpt < n:
                n = lpt
            if n > budget:
                n = budget
            target = r.num_computed + n
            ok = kv.allocate(r, target)
            while not ok:
                victim = running.pop()
                self._preempt(victim)
                preempted = True
                batch.num_preempted += 1
                if victim is r:
                    break
                ok = kv.allocate(r, target)
            if not ok:
                break
            self._commit(batch, r, n, now)
            budget -= n
            i += 1

        # Then admit waiting requests, FCFS, unless this step preempted.
        if not preempted:
            watermark = int(cfg.watermark * kv.num_blocks)
            while waiting and budget > 0 and len(running) < cfg.max_num_seqs:
                r = waiting[0]
                n = r.prompt_len + r.num_generated - r.num_computed
                if 0 < lpt < n:
                    n = lpt
                if not cfg.enable_chunked_prefill and n > budget:
                    break
                if n > budget:
                    n = budget
                keep = watermark if running else 0
                if cfg.reserve_full_isl and kv.needed(r, r.prompt_len + r.num_generated) + keep > (
                    kv.free
                ):
                    break
                if not kv.allocate(r, r.num_computed + n, keep):
                    break
                waiting.popleft()
                r.state = RUNNING
                running.append(r)
                self._commit(batch, r, n, now)
                budget -= n
        return batch


class PrefillFirstScheduler(Scheduler):
    """vLLM V0 default: prefill-only iterations first, decode iterations otherwise."""

    name = "prefill-first"

    def __init__(self, config: SchedulerConfig) -> None:
        if config.max_num_batched_tokens < config.max_model_len:
            raise ValueError("prefill-first needs max_num_batched_tokens >= max_model_len")
        super().__init__(config)

    def schedule(self, now: float) -> Batch:
        cfg, kv = self.cfg, self.kv
        batch = Batch()
        running, waiting = self.running, self.waiting
        watermark = int(cfg.watermark * kv.num_blocks)

        budget = cfg.max_num_batched_tokens
        while waiting and len(running) < cfg.max_num_seqs:
            r = waiting[0]
            n = r.prompt_len + r.num_generated  # recompute re-prefills generated tokens too
            keep = watermark if running else 0  # an idle engine must be able to admit
            if n > budget or kv.needed(r, n) + keep > kv.free:
                break
            kv.allocate(r, n)
            waiting.popleft()
            r.state = RUNNING
            running.append(r)
            self._commit(batch, r, n, now)
            budget -= n
        if batch:
            return batch

        # Decode iteration: every running request generates one token.
        i = 0
        while i < len(running):
            r = running[i]
            if r.done:
                i += 1
                continue
            target = r.num_computed + 1
            ok = kv.allocate(r, target)
            while not ok:
                victim = running.pop()
                self._preempt(victim)
                batch.num_preempted += 1
                if victim is r:
                    break
                ok = kv.allocate(r, target)
            if not ok:
                break
            self._commit(batch, r, 1, now)
            i += 1
        return batch


class OrcaScheduler(Scheduler):
    """Iteration-level batching with KV reserved for prompt + max_tokens at admission."""

    name = "orca"

    def _max_footprint(self, req: Request) -> int:
        return self._reserve(req)

    def schedule(self, now: float) -> Batch:
        cfg, kv = self.cfg, self.kv
        batch = Batch()
        running, waiting = self.running, self.waiting
        budget = cfg.max_num_batched_tokens
        for r in running:
            if not r.done:
                self._commit(batch, r, 1, now)
                budget -= 1
        while waiting and len(running) < cfg.max_num_seqs:
            r = waiting[0]
            n = r.prompt_len
            if n > budget and batch:
                break
            if not kv.allocate(r, self._reserve(r)):
                break
            waiting.popleft()
            r.state = RUNNING
            running.append(r)
            self._commit(batch, r, n, now)
            budget -= n
        return batch


class StaticScheduler(Scheduler):
    """Request-level (static) batching with padding.

    A batch launches when ``max_num_seqs`` requests wait, or when the oldest has waited
    ``max_wait`` seconds. The prompt step is padded to the longest prompt and every decode
    step runs all slots until the longest output is done; the batch's results are returned
    together when it completes, so ``finish`` is the batch end for every member.
    """

    name = "static"

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__(config)
        self._step = -1  # -1: no active batch; 0: prefill next; k>0: decode step k next
        self._max_prompt = 0
        self._max_output = 0

    def _max_footprint(self, req: Request) -> int:
        return self._reserve(req)

    def next_wakeup(self, now: float) -> float:
        if self._step < 0 and self.waiting:
            return self.waiting[0].arrival + self.cfg.max_wait
        return math.inf

    def _form_batch(self, now: float) -> bool:
        cfg, kv = self.cfg, self.kv
        if len(self.waiting) < cfg.max_num_seqs and (
            not self.waiting or now < self.waiting[0].arrival + cfg.max_wait
        ):
            return False
        members: list[Request] = []
        max_p = max_r = 0
        for r in self.waiting:
            if len(members) == cfg.max_num_seqs:
                break
            p, res = max(max_p, r.prompt_len), max(max_r, self._reserve(r) - r.prompt_len)
            # Padded batch: every slot holds max prompt + max reserved output.
            if kv.blocks_for(p + res) * (len(members) + 1) > kv.num_blocks:
                break
            members.append(r)
            max_p, max_r = p, res
        for r in members:
            self.waiting.popleft()
            r.state = RUNNING
        per_slot = kv.blocks_for(max_p + max_r)
        for r in members:
            kv.allocate(r, per_slot * kv.block_size)
        self.running = members
        self._max_prompt = max_p
        self._max_output = max(r.output_len for r in members)
        self._step = 0
        return True

    def schedule(self, now: float) -> Batch:
        batch = Batch()
        if self._step < 0 and not self._form_batch(now):
            return batch
        reqs = self.running
        s = batch.stats
        if self._step == 0:
            n = self._max_prompt
            for r in reqs:
                batch.reqs.append(r)
                batch.tokens.append(n)
                s.add(n, 0)
                r.first_scheduled = now
                r.num_computed = r.prompt_len
                r.num_generated = 1
                batch.sampled.append(r)
        else:
            ctx = self._max_prompt + self._step - 1
            for r in reqs:
                batch.reqs.append(r)
                batch.tokens.append(1)
                s.add(1, ctx)
                if r.num_generated < r.output_len:
                    r.num_computed += 1
                    r.num_generated += 1
                    batch.sampled.append(r)
        self._step += 1
        return batch

    def complete(self, batch: Batch, t: float) -> list[Request]:
        for r in batch.sampled:
            r.emitted += 1
            if r.emitted == 1:
                r.first_token = t
            else:
                gap = t - r.last_token
                r.max_itl = max(r.max_itl, gap)
                if r.itl is not None:
                    r.itl.append(gap)
            r.last_token = t
        if self._step > self._max_output - 1:  # all decode steps for the longest output ran
            done = self.running
            for r in done:
                r.finish = t
                r.state = FINISHED
                self.kv.release(r)
            self.running = []
            self._step = -1
            return done
        return []


def make_scheduler(name: str, config: SchedulerConfig) -> Scheduler:
    try:
        cls = Scheduler.registry[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(Scheduler.registry)}") from None
    return cls(config)


POLICIES = tuple(Scheduler.registry)
