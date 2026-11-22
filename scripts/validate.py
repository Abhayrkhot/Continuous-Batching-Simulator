"""Calibrate on the calibration replay, then predict every validation replay.

    python scripts/validate.py --profile qwen7b --model-name h200-qwen2.5-7b

Reads results/h200/<profile>/<config>-<job>/ run directories (copied from the cluster),
fits the step model on the calibration window of the run directory that has one, writes
the packaged model to src/batchsim/data/<model-name>.json, and writes
results/h200/<profile>/validation.json with predicted vs measured metrics for every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from batchsim.calibrate import fit_linear, periods
from batchsim.cost import RooflineModel, StepTimeModel
from batchsim.engine import EngineConfig, simulate
from batchsim.hardware import get_gpu, get_model
from batchsim.validation import (
    RunDir,
    compare,
    estimate_frontend,
    load_replay,
    step_model_error,
    validate_run,
)

MODEL_SHAPES = {"qwen7b": "qwen2.5-7b", "qwen32b": "qwen2.5-32b"}
ABLATION_KEYS = ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50", "e2e_p99")


def ablation(
    rd: RunDir, run: str, variants: dict[str, tuple[StepTimeModel, EngineConfig]]
) -> dict[str, dict[str, float]]:
    """Relative error of key metrics when one modeling choice is removed."""
    meas = load_replay(rd.path / f"{run}.jsonl")
    wl = meas.workload(run)
    real = meas.as_result()
    span = wl.duration
    out = {}
    for name, (cost, eng) in variants.items():
        pred = simulate(wl, cost, "vllm", rd.scheduler_config(), eng)
        cmp = compare(real, pred, span * 0.1, span * 0.1)
        out[name] = {k: cmp[k]["rel_err"] for k in ABLATION_KEYS if k in cmp}
    return out


def group_label(rd: RunDir) -> str:
    sc = rd.scheduler_config()
    label = f"budget {sc.max_num_batched_tokens}, {sc.max_num_seqs} seqs"
    if rd.args.get("num_gpu_blocks_override"):
        label += f", {rd.args['num_gpu_blocks_override']} blocks"
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--results", type=Path, default=Path("results/h200"))
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare with the committed validation.json instead of writing",
    )
    args = ap.parse_args()

    root = args.results / args.profile
    dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    runs = {d: RunDir.load(d) for d in dirs}
    # Calibrate on the most complete calibration replay.
    with_cal = [d for d, rd in runs.items() if "calibration" in rd.runs()]
    cal_dir = min(with_cal, key=lambda d: load_replay(d / "calibration.jsonl").errors)
    cal = runs[cal_dir]
    steps = cal.steps_in("calibration")
    y = periods(steps)
    stats = [s.stats for s in steps]

    rng = np.random.default_rng(0)
    test = rng.random(len(y)) < args.holdout
    m_tr, _ = fit_linear([s for s, t in zip(stats, test) if not t], y[~test])
    holdout = step_model_error(m_tr, [s for s, t in zip(steps, test) if t], y[test])
    model, fit = fit_linear(stats, y)

    fe = estimate_frontend(cal, model)
    delta, idle = fe["frontend_latency"], fe["idle_launch_latency"]
    model.meta = {
        "gpu": "NVIDIA H200 (141 GB)",
        "engine": f"vLLM {cal.server.get('vllm_version', '?')}",
        "model": cal.meta["model"],
        "calibration_run": cal_dir.name,
        "calibration_steps": len(y),
        "holdout_step_error": holdout,
        "fit": fit.to_dict(),
        "frontend_latency_s": delta,
        "idle_launch_latency_s": idle,
        "frontend_estimate": fe,
    }
    out_model = Path("src/batchsim/data") / f"{args.model_name}.json"
    if not args.check:
        model.save(out_model)
    print(
        f"step model: {len(y)} calibration iterations, held-out median error "
        f"{holdout['median_ape']:.1%}, mean {holdout['mape']:.1%}; frontend latency "
        f"{delta * 1e3:.1f} ms -> {out_model}"
    )

    full = EngineConfig(frontend_latency=delta, idle_launch_latency=idle)
    roofline = RooflineModel(get_model(MODEL_SHAPES[args.profile]), get_gpu("h200"))
    variants = {
        "full model": (model, full),
        "no async scheduling": (
            model,
            EngineConfig(async_scheduling=False, frontend_latency=delta, idle_launch_latency=idle),
        ),
        "no frontend latency": (model, EngineConfig()),
        "roofline step model": (roofline, full),
    }
    report: dict[str, Any] = {"profile": args.profile, "model": model.to_dict(), "configs": {}}
    seen: dict[tuple[str, str], Path] = {}
    for d, rd in runs.items():
        cfg = group_label(rd)
        group = report["configs"].setdefault(
            cfg,
            {
                "dirs": [],
                "scheduler": rd.scheduler_config().__dict__,
                "server": rd.server,
                "runs": [],
            },
        )
        group["dirs"].append(d.name)
        rows = group["runs"]
        for run in rd.runs():
            if run == "calibration":
                continue
            if (cfg, run) in seen:  # a later job repeated this replay; keep the later one
                rows[:] = [r for r in rows if r["run"] != run]
            seen[(cfg, run)] = d
            row = validate_run(rd, run, model, frontend_latency=delta, idle_launch_latency=idle)
            row["dir"] = d.name
            row["step_model"] = step_model_error(model, rd.steps_in(run))
            row["ablation"] = ablation(rd, run, variants)
            rows.append(row)
            m = row["metrics"]
            print(
                f"{cfg:>24} {run:>10}: TTFT p50 {m['ttft_p50']['real'] * 1e3:7.0f} vs "
                f"{m['ttft_p50']['sim'] * 1e3:7.0f} ms ({m['ttft_p50']['rel_err']:+.1%}), "
                f"p99 {m['ttft_p99']['rel_err']:+.1%}; TPOT p50 "
                f"{m['tpot_p50']['real'] * 1e3:5.1f} vs {m['tpot_p50']['sim'] * 1e3:5.1f} ms "
                f"({m['tpot_p50']['rel_err']:+.1%}), p99 {m['tpot_p99']['rel_err']:+.1%}; "
                f"steps {row['step_model'].get('median_ape', float('nan')):.1%}"
            )
    if args.check:
        sys.exit(check(report, json.loads((root / "validation.json").read_text())))
    (root / "validation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {root / 'validation.json'}")


def check(new: dict[str, Any], old: dict[str, Any], rtol: float = 1e-6) -> int:
    """0 if every recomputed metric matches the committed one."""
    bad = 0
    for cfg, c in old["configs"].items():
        fresh = {r["run"]: r for r in new["configs"][cfg]["runs"]}
        for row in c["runs"]:
            for k, m in row["metrics"].items():
                for side in ("real", "sim"):
                    a, b = m[side], fresh[row["run"]]["metrics"][k][side]
                    if abs(a - b) > rtol * max(abs(a), 1e-12):
                        print(f"MISMATCH {cfg}/{row['run']} {k}.{side}: {a} != {b}")
                        bad += 1
    print("validation reproduces" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    main()
