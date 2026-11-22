"""Model shapes and GPU specs: KV-cache sizing and the inputs to the roofline step model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1 << 30


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a dense decoder-only transformer (Llama/Qwen family)."""

    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    dtype_bytes: int = 2
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], name: str = "model") -> ModelConfig:
        heads = int(cfg["num_attention_heads"])
        hidden = int(cfg["hidden_size"])
        dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
        return cls(
            name=name,
            num_layers=int(cfg["num_hidden_layers"]),
            hidden_size=hidden,
            num_heads=heads,
            num_kv_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(cfg.get("head_dim") or hidden // heads),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            dtype_bytes=4 if "float32" in dtype else 2,
            tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        return cls.from_hf_config(json.loads(path.read_text()), name=path.parent.name)

    @property
    def layer_params(self) -> int:
        h, hd = self.hidden_size, self.head_dim
        attn = h * self.num_heads * hd * 2 + h * self.num_kv_heads * hd * 2  # q, o, k, v
        mlp = 3 * h * self.intermediate_size  # gate, up, down
        return self.num_layers * (attn + mlp)

    @property
    def lm_head_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def num_params(self) -> int:
        embed = self.vocab_size * self.hidden_size
        return self.layer_params + embed + (0 if self.tie_embeddings else self.lm_head_params)

    @property
    def weight_bytes(self) -> int:
        return self.num_params * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_bytes: int
    mem_bandwidth: float  # bytes/s
    peak_flops: float  # dense bf16 FLOP/s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODELS: dict[str, ModelConfig] = {
    m.name: m
    for m in (
        ModelConfig("qwen2.5-0.5b", 24, 896, 14, 2, 64, 4864, 151936, tie_embeddings=True),
        ModelConfig("qwen2.5-7b", 28, 3584, 28, 4, 128, 18944, 152064),
        ModelConfig("qwen2.5-32b", 64, 5120, 40, 8, 128, 27648, 152064),
        ModelConfig("llama-3.1-8b", 32, 4096, 32, 8, 128, 14336, 128256),
        ModelConfig("llama-3.1-70b", 80, 8192, 64, 8, 128, 28672, 128256),
    )
}

GPUS: dict[str, GPUSpec] = {
    g.name: g
    for g in (
        GPUSpec("h200", 141 * 10**9, 4.8e12, 989e12),
        GPUSpec("h100", 80 * 10**9, 3.35e12, 989e12),
        GPUSpec("a100-80g", 80 * 10**9, 2.039e12, 312e12),
        GPUSpec("b200", 180 * 10**9, 8.0e12, 2250e12),
        GPUSpec("l40s", 48 * 10**9, 0.864e12, 362e12),
    )
}


def get_model(name: str) -> ModelConfig:
    try:
        return MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}") from None


def get_gpu(name: str) -> GPUSpec:
    try:
        return GPUS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown GPU {name!r}; known: {sorted(GPUS)}") from None


def kv_cache_blocks(
    model: ModelConfig,
    gpu: GPUSpec,
    *,
    gpu_memory_utilization: float = 0.9,
    block_size: int = 16,
    tensor_parallel: int = 1,
    reserved_bytes: int = 4 * GIB,
) -> int:
    """KV-cache blocks per engine, the way vLLM sizes them.

    vLLM gives the cache what is left of ``gpu_memory_utilization`` of device memory
    after weights and the profiled activation peak (``reserved_bytes`` here; CUDA graphs
    and the sampler add a few GiB). Pass the engine's logged block count instead when
    you have it: it is exact.
    """
    budget = gpu.memory_bytes * gpu_memory_utilization * tensor_parallel
    free = budget - model.weight_bytes - reserved_bytes * tensor_parallel
    if free <= 0:
        raise ValueError(f"{model.name} does not fit on {tensor_parallel}x {gpu.name}")
    return int(free // (block_size * model.kv_bytes_per_token))
