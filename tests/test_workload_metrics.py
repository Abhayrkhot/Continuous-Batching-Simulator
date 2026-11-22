"""Workloads, metrics and capacity search."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from batchsim.engine import simulate
from batchsim.hardware import ModelConfig, get_gpu, get_model, kv_cache_blocks
from batchsim.metrics import SLO, weighted_percentile
from batchsim.request import RequestSpec
from batchsim.schedulers import SchedulerConfig
from batchsim.sweep import capacity, max_rate, rate_sweep
from batchsim.workload import (
    Empirical,
    Fixed,
    LogNormal,
    Uniform,
    Workload,
    arrivals,
    from_lengths,
    load_azure_trace,
    synthetic,
)
from tests.conftest import wl


@pytest.mark.parametrize("process,cv", [("poisson", 1.0), ("gamma", 3.0), ("uniform", 0.0)])
def test_arrival_processes(process: str, cv: float) -> None:
    t = arrivals(20_000, 5.0, np.random.default_rng(0), process, cv if process == "gamma" else 1.0)
    gaps = np.diff(t)
    assert t[0] == 0 and np.all(gaps >= 0)
    assert gaps.mean() == pytest.approx(0.2, rel=0.05)
    assert gaps.std() / gaps.mean() == pytest.approx(cv, abs=0.1)


def test_length_distributions() -> None:
    rng = np.random.default_rng(0)
    assert np.all(Fixed(7).sample(rng, 5) == 7)
    u = Uniform(3, 5).sample(rng, 1000)
    assert u.min() == 3 and u.max() == 5
    ln = LogNormal(500, 1.0, 1, 10_000).sample(rng, 50_000)
    assert ln.mean() == pytest.approx(500, rel=0.05) and ln.min() >= 1
    assert set(Empirical((1, 9)).sample(rng, 100)) <= {1, 9}


def test_scale_window_clip_csv(tmp_path: Path) -> None:
    w = synthetic(500, 4.0, LogNormal(300, 1.0), LogNormal(50, 1.0), process="gamma", cv=2.0)
    gaps = np.diff([s.arrival for s in w.specs])
    fast = w.scale_to_rate(40.0)
    fast_gaps = np.diff([s.arrival for s in fast.specs])
    assert fast.rate == pytest.approx(40.0)
    assert fast_gaps.std() / fast_gaps.mean() == pytest.approx(gaps.std() / gaps.mean())
    win = w.window(10.0, 20.0)
    assert all(0 <= s.arrival < 10 for s in win.specs)
    clipped = w.clip(256)
    assert max(s.prompt_len + s.output_len for s in clipped.specs) <= 256
    path = tmp_path / "w.csv"
    w.head(20).to_csv(path)
    back = Workload.from_csv(path)
    assert [(s.prompt_len, s.output_len) for s in back.specs] == [
        (s.prompt_len, s.output_len) for s in w.head(20).specs
    ]


def test_workload_validation() -> None:
    with pytest.raises(ValueError):
        RequestSpec(0.0, 0, 1)
    with pytest.raises(ValueError):
        RequestSpec(0.0, 1, 0)
    with pytest.raises(ValueError):
        RequestSpec(0.0, 1, 5, max_tokens=4)
    with pytest.raises(ValueError):
        Workload((RequestSpec(1.0, 1, 1), RequestSpec(0.0, 1, 1)))
    assert from_lengths([3, 4], [1, 2]).rate == math.inf


def test_azure_loader(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    path.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2023-11-16 18:15:46.6805900,374,44\n"
        "2023-11-16 18:15:50.9951690,396,0\n"
    )
    w = load_azure_trace(path)
    assert [s.arrival for s in w.specs] == pytest.approx([0.0, 4.314579])
    assert w.specs[1].output_len == 1  # zero-token generations become one token


def test_metrics_and_slo(unit_cost) -> None:
    res = simulate(wl((0.0, 4, 1), (0.0, 4, 5)), unit_cost, "vllm", SchedulerConfig(num_blocks=10))
    assert math.isnan(res.tpot[0])
    assert res.tpot[1] == 1.0
    assert res.slo_attainment(SLO(ttft=1.0)) == 1.0
    assert res.slo_attainment(SLO(e2e=1.0)) == 0.5
    s = res.summary(slo=SLO(ttft=1.0, tpot=0.5))
    assert s["slo_attainment"] == 0.5  # the single-token request has no TPOT to miss
    assert str(SLO(ttft=0.2, tpot=0.05)) == "TTFT<=200ms & TPOT<=50ms"


def test_weighted_percentile() -> None:
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([1.0, 2.0, 1.0])
    expanded = np.repeat(v, w.astype(int))
    for q in (25, 50, 75, 100):
        assert weighted_percentile(v, w, [q])[0] == np.percentile(
            expanded, q, method="inverted_cdf"
        )


def test_max_rate_bisection() -> None:
    assert max_rate(lambda r: r <= 7.3, 1.0, 100.0, rel_tol=0.001) == pytest.approx(7.3, rel=0.002)
    assert max_rate(lambda r: False, 1.0, 10.0) == 0.0
    assert max_rate(lambda r: True, 1.0, 10.0) == 10.0


def test_capacity_and_sweep(token_cost) -> None:
    base = synthetic(400, 1.0, LogNormal(200, 0.5), LogNormal(40, 0.5), seed=3)
    cfg = SchedulerConfig(num_blocks=5000, max_num_batched_tokens=512)
    loose = capacity(base, token_cost, SLO(ttft=2.0), scheduler=cfg, hi=500)
    tight = capacity(base, token_cost, SLO(ttft=0.1), scheduler=cfg, hi=500)
    assert 0 < tight["rate"] < loose["rate"]
    rows = rate_sweep(base, [1.0, 50.0], token_cost, scheduler=cfg)
    assert rows[1]["ttft_p99"] > rows[0]["ttft_p99"]


def test_hardware_and_kv_sizing(tmp_path: Path) -> None:
    m = get_model("qwen2.5-7b")
    assert m.kv_bytes_per_token == 2 * 28 * 4 * 128 * 2
    assert 7.0e9 < m.num_params < 8.0e9
    blocks = kv_cache_blocks(m, get_gpu("h200"))
    assert 100_000 < blocks < 140_000
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, '
        '"num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100, '
        '"torch_dtype": "bfloat16"}'
    )
    tiny = ModelConfig.from_json(cfg)
    assert tiny.head_dim == 16 and tiny.kv_bytes_per_token == 2 * 2 * 2 * 16 * 2
    with pytest.raises(ValueError):
        get_model("gpt-17")
    with pytest.raises(ValueError):
        kv_cache_blocks(get_model("llama-3.1-70b"), get_gpu("l40s"))
