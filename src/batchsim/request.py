"""Request specs (immutable workload entries) and their mutable simulation state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NAN = math.nan


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """One request in a workload: when it arrives and how many tokens it moves.

    ``output_len`` is the number of tokens the request generates (benchmarks run with
    ``ignore_eos`` so this is exact). ``max_tokens`` is the cap the client declared, which
    reservation-based policies (static batching, Orca) must provision for; it defaults to
    ``output_len``.
    """

    arrival: float
    prompt_len: int
    output_len: int
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_len < 1:
            raise ValueError(f"prompt_len must be >= 1, got {self.prompt_len}")
        if self.output_len < 1:
            raise ValueError(f"output_len must be >= 1, got {self.output_len}")
        if self.max_tokens is not None and self.max_tokens < self.output_len:
            raise ValueError("max_tokens must be >= output_len")
        if not math.isfinite(self.arrival) or self.arrival < 0:
            raise ValueError(f"arrival must be finite and >= 0, got {self.arrival}")

    @property
    def reserve_tokens(self) -> int:
        return self.prompt_len + (self.max_tokens or self.output_len)


@dataclass(slots=True, eq=False)
class Request:
    """Simulation state of one request.

    Token accounting follows vLLM's V1 engine: ``num_tokens`` is prompt plus generated
    tokens, ``num_computed`` is how many of them have KV in the cache. A request needs
    ``num_tokens - num_computed`` more tokens computed before it can sample its next token,
    so a decoding request needs exactly one. Preemption by recompute resets
    ``num_computed`` to zero, after which prompt *and* generated tokens are prefilled again.
    """

    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    reserve_tokens: int
    # Whether the client declared max_tokens (reservation policies then provision for it).
    declared_max_tokens: bool = False
    num_computed: int = 0
    num_generated: int = 0
    num_blocks: int = 0
    num_preemptions: int = 0
    # Tokens delivered to the client (num_generated also counts in-flight placeholders).
    emitted: int = 0
    # 0 waiting, 1 running, 2 finished (see batchsim.schedulers).
    state: int = 0
    # Set once the request's last token is scheduled; it still holds its KV until the step
    # that produces the token completes.
    done: bool = False
    first_scheduled: float = NAN
    first_token: float = NAN
    last_token: float = NAN
    finish: float = NAN
    max_itl: float = 0.0
    itl: list[float] | None = field(default=None, repr=False)

    @classmethod
    def from_spec(cls, rid: int, spec: RequestSpec) -> Request:
        return cls(
            rid=rid,
            arrival=spec.arrival,
            prompt_len=spec.prompt_len,
            output_len=spec.output_len,
            reserve_tokens=spec.reserve_tokens,
            declared_max_tokens=spec.max_tokens is not None,
        )

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def remaining(self) -> int:
        """Tokens to compute before this request can sample its next token."""
        return self.prompt_len + self.num_generated - self.num_computed

    @property
    def is_decode(self) -> bool:
        return self.num_generated > 0 and self.remaining == 1
