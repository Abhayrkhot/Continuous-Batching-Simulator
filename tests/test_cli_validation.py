"""CLI commands and the real-engine validation pipeline on synthetic run directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchsim.cli import main
from batchsim.cost import LinearStepModel
from batchsim.engine import EngineConfig, simulate
from batchsim.validation import (
    RunDir,
    estimate_frontend,
    load_replay,
    parse_server_log,
    parse_vllm_args,
    step_model_error,
    validate_run,
)
from batchsim.workload import LogNormal, synthetic

COMMON = ["--workload", "synthetic", "--n", "300", "--num-blocks", "20000"]


def test_cli_simulate_compare_sweep(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["simulate", *COMMON, "--rate", "5", "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "TTFT p50 ms" in out and "roofline step model" in out
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["requests"] == 300
    assert main(["compare", *COMMON, "--rate", "5", "--slo-ttft", "500"]) == 0
    out = capsys.readouterr().out
    for pol in ("vllm", "prefill-first", "orca", "static"):
        assert pol in out
    assert main(["sweep", *COMMON, "--rates", "2", "8"]) == 0


def test_cli_capacity_and_workload(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["capacity", *COMMON, "--slo-ttft", "1000", "--budgets", "512", "2048"]) == 0
    assert "budget=512" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["capacity", *COMMON])
    out = tmp_path / "w.csv"
    assert (
        main(
            ["workload", "--workload", "synthetic", "--n", "300", "--rate", "3", "--out", str(out)]
        )
        == 0
    )
    assert out.read_text().startswith("arrival,prompt_len,output_len")


def test_cli_uses_calibrated_latencies(tmp_path: Path) -> None:
    from batchsim.cli import build_parser, engine_config

    m = LinearStepModel.simple(const=0.004, per_token=2e-5)
    m.meta = {"frontend_latency_s": 0.003, "idle_launch_latency_s": 0.001}
    args = build_parser().parse_args(["simulate"])
    eng = engine_config(args, m)
    assert (eng.frontend_latency, eng.idle_launch_latency) == (0.003, 0.001)
    args = build_parser().parse_args(["simulate", "--frontend-latency", "0"])
    assert engine_config(args, m).frontend_latency == 0.0


def test_cli_cost_model_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "m.json"
    LinearStepModel.simple(const=0.004, per_token=2e-5).save(path)
    assert main(["simulate", *COMMON, "--rate", "5", "--cost", str(path)]) == 0
    assert "linear step model" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["simulate", *COMMON, "--cost", "no-such-model"])


def test_parsers(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "INFO [core.py:153] Initializing a V1 LLM engine (v0.34.0) with config\n"
        "INFO [kv_cache_utils.py:1168] GPU KV cache size: 2,022,400 tokens, Maximum concurrency\n"
    )
    assert parse_server_log(log) == {
        "kv_tokens": 2022400,
        "num_blocks": 126399,
        "vllm_version": "0.34.0",
    }
    assert parse_vllm_args("--max-num-batched-tokens 2048 --max-num-seqs 256") == {
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 256,
    }
    assert parse_vllm_args("")["max_num_batched_tokens"] == 8192
    assert parse_vllm_args("--max-num-seqs=64 --max-model-len 4096") == {
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 64,
        "max_model_len": 4096,
    }


def _fake_run_dir(tmp_path: Path, cost: LinearStepModel, delta: float) -> Path:
    """A run directory whose 'measurements' are the simulator's own output plus delta."""
    d = tmp_path / "default-1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"config": "default", "model": "m", "vllm_args": ""}))
    (d / "server.log").write_text("GPU KV cache size: 160,016 tokens\n")
    t0 = 1.7e9
    runs = []
    for name, rate in (("calibration", 0.2), ("conv_r4", 4.0)):
        w = synthetic(60, rate, LogNormal(300, 0.5), LogNormal(30, 0.5), seed=7)
        res = simulate(
            w,
            cost,
            "vllm",
            RunDir(d, {}, {"num_blocks": 10000}, parse_vllm_args(""), {}).scheduler_config(),
            EngineConfig(),
        )
        with (d / f"{name}.jsonl").open("w") as f:
            for i, s in enumerate(w.specs):
                first = res.first_token[i] - s.arrival + delta
                last = res.finish[i] - s.arrival + delta
                toks = [
                    first + (last - first) * k / max(1, s.output_len - 1)
                    for k in range(s.output_len)
                ]
                rec = {
                    "i": i,
                    "send": t0 + s.arrival,
                    "prompt_len": s.prompt_len,
                    "output_len": s.output_len,
                    "tok": toks,
                    "completion_tokens": s.output_len,
                }
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps({"summary": {"requests": len(w)}}) + "\n")
        runs.append({"run": name, "start": t0, "end": t0 + w.duration + 60})
        t0 += 10_000
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    (d / "steps.jsonl").write_text("")
    return d


def test_validation_pipeline_recovers_itself(tmp_path: Path) -> None:
    cost = LinearStepModel.simple(const=0.005, per_token=2e-5)
    d = _fake_run_dir(tmp_path, cost, delta=0.004)
    rd = RunDir.load(d)
    assert rd.runs() == ["calibration", "conv_r4"]
    assert rd.scheduler_config().num_blocks == 160016 // 16 - 1
    fe = estimate_frontend(rd, cost)
    delta = fe["frontend_latency"]
    assert delta == pytest.approx(0.004, abs=1e-6)
    assert fe["idle_launch_latency"] == pytest.approx(0.0, abs=1e-6)
    row = validate_run(rd, "conv_r4", cost, frontend_latency=delta)
    assert row["errors"] == 0
    for m in ("ttft_p50", "ttft_p99", "tpot_p50", "e2e_p99"):
        assert abs(row["metrics"][m]["rel_err"]) < 1e-4, m  # wall-clock float rounding
    meas = load_replay(d / "conv_r4.jsonl")
    assert len(meas.send) == 60
    assert step_model_error(cost, []) == {}


def test_compacted_results_validate_identically(tmp_path: Path) -> None:
    import subprocess
    import sys

    cost = LinearStepModel.simple(const=0.005, per_token=2e-5)
    d = _fake_run_dir(tmp_path, cost, delta=0.004)
    before = validate_run(RunDir.load(d), "conv_r4", cost)
    script = Path(__file__).resolve().parents[1] / "scripts" / "compact_results.py"
    subprocess.run([sys.executable, str(script), str(d)], check=True, capture_output=True)
    assert (d / "conv_r4.jsonl.gz").exists() and not (d / "conv_r4.jsonl").exists()
    assert (d / "server.log.gz").exists()
    after = validate_run(RunDir.load(d), "conv_r4", cost)
    assert after["metrics"] == before["metrics"]
