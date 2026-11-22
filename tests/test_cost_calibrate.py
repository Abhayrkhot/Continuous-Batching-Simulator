"""Step-time models and the calibration fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from batchsim.calibrate import (
    calibrate_from_probe,
    evaluate,
    fit_linear,
    load_probe,
    nnls,
    periods,
    stats_from_probe,
)
from batchsim.cost import (
    VLLM_CAPTURE_SIZES,
    BatchStats,
    LinearStepModel,
    RooflineModel,
    load_cost_model,
    pad_tokens,
)
from batchsim.hardware import get_gpu, get_model


def test_batch_stats_accounting() -> None:
    s = BatchStats()
    s.add(1, 99)  # decode reading 100 keys
    s.add(4, 10)  # prefill chunk: 4*10 + 4*5/2 pairs
    assert (s.num_tokens, s.num_seqs, s.num_decode, s.num_prefill) == (5, 2, 1, 1)
    assert s.decode_ctx == 100
    assert s.prefill_attn == 4 * 10 + 10
    assert s.copy() == s


def test_pad_tokens() -> None:
    assert pad_tokens(1, VLLM_CAPTURE_SIZES) == 1
    assert pad_tokens(3, VLLM_CAPTURE_SIZES) == 4
    assert pad_tokens(9, VLLM_CAPTURE_SIZES) == 16
    assert pad_tokens(257, VLLM_CAPTURE_SIZES) == 272
    assert pad_tokens(513, VLLM_CAPTURE_SIZES) == 513
    assert len(VLLM_CAPTURE_SIZES) == 51


def test_roofline_regimes() -> None:
    m, g = get_model("qwen2.5-7b"), get_gpu("h200")
    rf = RooflineModel(m, g, overhead_s=0.0, per_seq_s=0.0)
    one = BatchStats()
    one.add(1, 0)
    weights_time = (m.layer_params + m.lm_head_params) * 2 / (g.mem_bandwidth * rf.mbu)
    assert rf.step_time(one) == pytest.approx(weights_time, rel=1e-3)  # memory bound
    big = BatchStats()
    big.add(8192, 0)
    assert rf.step_time(big) > 10 * rf.step_time(one)  # compute bound
    longer = BatchStats()
    longer.add(1, 100_000)
    assert rf.step_time(longer) > rf.step_time(one)


@pytest.mark.parametrize(
    "model",
    [
        LinearStepModel.simple(const=0.002, per_token=3e-5, seqs=1e-5),
        RooflineModel(get_model("llama-3.1-8b"), get_gpu("a100-80g")),
    ],
)
def test_cost_model_roundtrip(model, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    model.save(path)
    back = load_cost_model(path)
    s = BatchStats()
    s.add(300, 1000)
    s.add(1, 5000)
    assert back.step_time(s) == pytest.approx(model.step_time(s))


def test_linear_model_requires_all_coefficients() -> None:
    with pytest.raises(ValueError, match="missing"):
        LinearStepModel(coef={"const": 1.0})


def test_nnls_matches_scipy() -> None:
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(40, 6)) * rng.uniform(0.1, 100, size=6)
        b = rng.normal(size=40)
        ours = nnls(a, b)
        ref, _ = scipy_opt.nnls(a, b)
        assert np.all(ours >= 0)
        assert np.linalg.norm(a @ ours - b) == pytest.approx(np.linalg.norm(a @ ref - b), rel=1e-6)


def _random_stats(rng: np.random.Generator, n: int) -> list[BatchStats]:
    out = []
    for _ in range(n):
        s = BatchStats()
        for _ in range(rng.integers(0, 64)):
            s.add(1, int(rng.integers(0, 8000)))
        for _ in range(rng.integers(0, 3)):
            s.add(int(rng.integers(2, 2048)), int(rng.integers(0, 4000)))
        if s.num_seqs == 0:
            s.add(1, 10)
        out.append(s)
    return out


def test_fit_recovers_a_known_model() -> None:
    truth = LinearStepModel(
        coef={
            "const": 3e-3,
            "tokens": 2e-6,
            "seqs": 2e-5,
            "prefill_seqs": 1e-4,
            "prefill_attn_m": 4e-4,
            "decode_ctx_m": 6e-3,
            "hinge_512": 3e-5,
        },
        knots=(512,),
    )
    rng = np.random.default_rng(1)
    stats = _random_stats(rng, 3000)
    y = np.array([truth.step_time(s) for s in stats]) * rng.normal(1.0, 0.01, size=len(stats))
    model, rep = fit_linear(stats, y, knots=(512,))
    assert rep.median_ape < 0.01
    test = _random_stats(np.random.default_rng(2), 500)
    y_test = np.array([truth.step_time(s) for s in test])
    assert evaluate(model, test, y_test).mape < 0.01


def _probe_lines(n: int, idle_every: int = 50) -> list[dict[str, object]]:
    truth = LinearStepModel.simple(const=4e-3, per_token=1e-5, decode_ctx_m=5e-3)
    rng = np.random.default_rng(3)
    t = 1000.0
    lines = []
    for i in range(n):
        dec = int(rng.integers(1, 100))
        ctx = int(rng.integers(dec, dec * 4000))
        pre = [[int(rng.integers(2, 900)), int(rng.integers(0, 3000))]] if i % 3 == 0 else []
        rec = {
            "i": i,
            "cpu": t - 0.005,
            "T": dec + sum(p[0] for p in pre),
            "S": dec + len(pre),
            "D": dec,
            "Dc": ctx,
            "p": pre,
        }
        dur = truth.step_time(stats_from_probe(rec))
        if i % idle_every == 0:
            t += 0.5  # idle gap: the next step starts on its own
        rec["g0"], rec["g1"] = t, t + dur
        t += dur
        lines.append(rec)
    return lines


def test_periods_and_probe_parsing(tmp_path: Path) -> None:
    lines = _probe_lines(10, idle_every=4)
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    steps = load_probe(path)
    assert [s.index for s in steps] == list(range(10))
    y = periods(steps)
    np.testing.assert_allclose(y, [s.g1 - s.g0 for s in steps])
    s0 = steps[3].stats
    assert s0.decode_ctx == lines[3]["Dc"] + lines[3]["D"]  # type: ignore[operator]


def test_calibrate_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _probe_lines(2000)))
    model, report = calibrate_from_probe(path)
    assert report["holdout"]["median_ape"] < 0.01
    assert report["steps"] == 2000
    assert model.meta["fit"]["holdout"]["mape"] < 0.02
