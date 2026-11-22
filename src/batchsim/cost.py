"""Step-time models: how long one engine iteration takes for a given batch.

Every scheduler builds a :class:`BatchStats` for each iteration it runs; a
:class:`StepTimeModel` turns it into seconds. Two models ship:

* :class:`RooflineModel` needs only the model shape and GPU datasheet numbers. It is the
  prior for hardware you have not measured.
* :class:`LinearStepModel` is a grey-box model fitted (``batchsim calibrate``) to per-step
  GPU timings recorded inside a real vLLM engine. Its features follow the cost structure
  of a transformer step: a convex piecewise-linear term in the (CUDA-graph padded) token
  count for the weight GEMMs, a per-sequence term for sampling and host work, a
  query-key-pair term for prefill attention and a KV-read term for decode attention.

Both models are affine in the context lengths for a fixed batch shape, which the engine
relies on to fast-forward runs of identical decode steps.
"""

from __future__ import annotations

import bisect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from batchsim.hardware import GPUSpec, ModelConfig

# vLLM 0.34 default CUDA graph capture sizes (CompilationConfig docs).
VLLM_CAPTURE_SIZES: tuple[int, ...] = (1, 2, 4, *range(8, 256, 8), *range(256, 512 + 1, 16))


@dataclass(slots=True)
class BatchStats:
    """Shape of one iteration.

    ``prefill_attn`` counts query-key pairs of the prefill chunks: a chunk of ``n`` new
    tokens after ``c`` cached ones attends to ``n*c + n*(n+1)/2`` pairs. ``decode_ctx`` is
    the number of cached tokens the decode queries read (context plus the new token).
    """

    num_tokens: int = 0
    num_seqs: int = 0
    num_decode: int = 0
    num_prefill: int = 0
    prefill_tokens: int = 0
    prefill_attn: float = 0.0
    decode_ctx: float = 0.0

    def add(self, new_tokens: int, context: int) -> None:
        """Add one sequence computing ``new_tokens`` after ``context`` cached tokens.

        One-token entries count as decodes, as in the probe that calibrates the model: the
        kernels cannot tell a one-token prefill chunk from a decode.
        """
        self.num_tokens += new_tokens
        self.num_seqs += 1
        if new_tokens == 1:
            self.num_decode += 1
            self.decode_ctx += context + new_tokens
        else:
            self.num_prefill += 1
            self.prefill_tokens += new_tokens
            self.prefill_attn += new_tokens * context + new_tokens * (new_tokens + 1) / 2

    def copy(self) -> BatchStats:
        return BatchStats(
            self.num_tokens,
            self.num_seqs,
            self.num_decode,
            self.num_prefill,
            self.prefill_tokens,
            self.prefill_attn,
            self.decode_ctx,
        )


class StepTimeModel(ABC):
    kind: ClassVar[str]
    registry: ClassVar[dict[str, type[StepTimeModel]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "kind", None):
            StepTimeModel.registry[cls.kind] = cls

    @abstractmethod
    def step_time(self, s: BatchStats) -> float:
        """Seconds for one iteration with batch shape ``s``."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> StepTimeModel: ...

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"kind": self.kind, **self.to_dict()}, indent=2) + "\n")


def load_cost_model(path: str | Path) -> StepTimeModel:
    d = json.loads(Path(path).read_text())
    kind = d.pop("kind")
    return StepTimeModel.registry[kind].from_dict(d)


def pad_tokens(n: int, sizes: Sequence[int]) -> int:
    """Round ``n`` up to the next CUDA graph capture size; beyond the largest, no padding."""
    if not sizes or n > sizes[-1]:
        return n
    return sizes[bisect.bisect_left(sizes, n)]


@dataclass
class RooflineModel(StepTimeModel):
    """Datasheet roofline: each part of the step is bound by FLOPs or by bytes moved.

    ``mfu``/``mbu`` are the achieved fractions of peak compute/bandwidth. The defaults are
    typical of bf16 dense models on Hopper and are deliberately not tuned to any
    measurement; calibrate a :class:`LinearStepModel` when accuracy matters.
    """

    kind: ClassVar[str] = "roofline"
    model: ModelConfig
    gpu: GPUSpec
    tensor_parallel: int = 1
    mfu: float = 0.55
    mbu: float = 0.75
    overhead_s: float = 1.5e-3
    per_seq_s: float = 4e-6

    def step_time(self, s: BatchStats) -> float:
        m, tp = self.model, self.tensor_parallel
        flops_peak = self.gpu.peak_flops * self.mfu * tp
        bw = self.gpu.mem_bandwidth * self.mbu * tp
        lin_flops = 2.0 * m.layer_params * s.num_tokens + 2.0 * m.lm_head_params * s.num_seqs
        lin_bytes = (m.layer_params + m.lm_head_params) * m.dtype_bytes
        t_lin = max(lin_flops / flops_peak, lin_bytes / bw)
        attn_flops = 4.0 * m.num_layers * m.num_heads * m.head_dim * s.prefill_attn
        kv_bytes = m.kv_bytes_per_token * s.decode_ctx
        return (
            self.overhead_s
            + self.per_seq_s * s.num_seqs
            + t_lin
            + attn_flops / flops_peak
            + kv_bytes / bw
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "gpu": self.gpu.to_dict(),
            "tensor_parallel": self.tensor_parallel,
            "mfu": self.mfu,
            "mbu": self.mbu,
            "overhead_s": self.overhead_s,
            "per_seq_s": self.per_seq_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RooflineModel:
        d = dict(d)
        model = ModelConfig(**d.pop("model"))
        gpu = GPUSpec(**d.pop("gpu"))
        return cls(model=model, gpu=gpu, **d)


LINEAR_FEATURES_BASE = ("const", "tokens", "seqs", "prefill_seqs", "prefill_attn_m", "decode_ctx_m")


@dataclass
class LinearStepModel(StepTimeModel):
    """Grey-box linear model with non-negative coefficients (fitted by NNLS).

    ``t = c0 + c_tok*T' + sum_k h_k*max(0, T'-k) + c_seq*S + c_pseq*P + c_pa*A + c_dc*D``

    where ``T'`` is the token count padded to the CUDA graph capture size, ``S`` sequences,
    ``P`` prefill sequences, ``A`` prefill query-key pairs (millions) and ``D`` decode KV
    tokens read (millions). Non-negative hinge weights make the token term convex: flat
    while weight loading dominates, then compute-bound.
    """

    kind: ClassVar[str] = "linear"
    coef: dict[str, float]
    knots: tuple[int, ...] = ()
    capture_sizes: tuple[int, ...] = VLLM_CAPTURE_SIZES
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.knots = tuple(self.knots)
        self.capture_sizes = tuple(self.capture_sizes)
        names = self.feature_names(self.knots)
        missing = [n for n in names if n not in self.coef]
        if missing:
            raise ValueError(f"missing coefficients: {missing}")
        self._w = [float(self.coef[n]) for n in names]

    @staticmethod
    def feature_names(knots: Sequence[int]) -> tuple[str, ...]:
        return LINEAR_FEATURES_BASE + tuple(f"hinge_{k}" for k in knots)

    @staticmethod
    def features(s: BatchStats, knots: Sequence[int], capture_sizes: Sequence[int]) -> list[float]:
        t = float(pad_tokens(s.num_tokens, capture_sizes))
        row = [
            1.0,
            t,
            float(s.num_seqs),
            float(s.num_prefill),
            s.prefill_attn * 1e-6,
            s.decode_ctx * 1e-6,
        ]
        row.extend(max(0.0, t - k) for k in knots)
        return row

    def step_time(self, s: BatchStats) -> float:
        x = self.features(s, self.knots, self.capture_sizes)
        return math.fsum(w * v for w, v in zip(self._w, x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": self.coef,
            "knots": list(self.knots),
            "capture_sizes": list(self.capture_sizes),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinearStepModel:
        return cls(
            coef={k: float(v) for k, v in d["coef"].items()},
            knots=tuple(d.get("knots", ())),
            capture_sizes=tuple(d.get("capture_sizes", VLLM_CAPTURE_SIZES)),
            meta=d.get("meta", {}),
        )

    @classmethod
    def simple(cls, const: float, per_token: float, **extra: float) -> LinearStepModel:
        """A hand-written model (tests, examples): ``const + per_token*T`` plus extras."""
        coef = dict.fromkeys(LINEAR_FEATURES_BASE, 0.0)
        coef.update(const=const, tokens=per_token, **extra)
        return cls(coef=coef, knots=(), capture_sizes=())
