"""Simulator-only experiments on the H200-calibrated step model.

    python scripts/experiments.py --cost h200-qwen2.5-7b --out results/h200/qwen7b/experiments.json

1. Policies: SLO attainment vs request rate, and goodput (highest rate with >= 90% of
   requests meeting the SLO) for static batching, Orca-style iteration-level batching,
   prefill-first paged batching (vLLM V0) and vLLM V1 chunked prefill, on the Azure traces.
2. Token budget: goodput of vLLM V1 across ``max_num_batched_tokens``.
3. Speed: wall time to simulate the full one-hour Azure conversation trace.

Every number comes from the simulator; the README marks which ones were also measured.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib import resources
from pathlib import Path
from typing import Any

from batchsim.cost import load_cost_model
from batchsim.engine import EngineConfig, simulate
from batchsim.metrics import SLO
from batchsim.schedulers import SchedulerConfig
from batchsim.sweep import capacity, rate_sweep
from batchsim.workload import azure

MAX_MODEL_LEN = 16384
SLOS = {
    # Chat: first token within half a second, then faster than reading speed.
    "conv": SLO(ttft=0.5, tpot=0.025),
    # Code completion: the whole (short) completion must land quickly.
    "code": SLO(ttft=0.5, e2e=1.5),
}


def configs(num_blocks: int) -> dict[str, tuple[str, SchedulerConfig]]:
    base = SchedulerConfig(
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        num_blocks=num_blocks,
        max_model_len=MAX_MODEL_LEN,
    )
    return {
        "static (batch 32)": ("static", base.with_(max_num_seqs=32, max_wait=0.05)),
        "static (batch 128)": ("static", base.with_(max_num_seqs=128, max_wait=0.05)),
        "orca (reserve 2048)": (
            "orca",
            base.with_(max_num_batched_tokens=1 << 30, reserve_max_tokens=2048),
        ),
        "orca (oracle)": ("orca", base.with_(max_num_batched_tokens=1 << 30)),
        "prefill-first": (
            "prefill-first",
            base.with_(max_num_batched_tokens=MAX_MODEL_LEN, watermark=0.01),
        ),
        "vllm v1 (budget 8192)": ("vllm", base),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost", default="h200-qwen2.5-7b")
    ap.add_argument("--num-blocks", type=int, default=126399)  # vLLM 0.34, Qwen2.5-7B, H200
    ap.add_argument("--requests", type=int, default=1500)
    ap.add_argument("--rate-scale", type=float, default=1.0, help="scale the rate grids")
    ap.add_argument("--slo-scale", type=float, default=1.0, help="scale every SLO (larger models)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    path = Path(args.cost)
    if not path.exists():
        path = Path(str(resources.files("batchsim") / "data" / f"{args.cost}.json"))
    cost = load_cost_model(path)
    meta = getattr(cost, "meta", {})
    eng = EngineConfig(
        async_scheduling=True,
        frontend_latency=meta.get("frontend_latency_s", 0.0),
        idle_launch_latency=meta.get("idle_launch_latency_s", 0.0),
        record_steps=False,
    )
    slos = {
        k: SLO(v.ttft * args.slo_scale, v.tpot * args.slo_scale, v.e2e * args.slo_scale)
        for k, v in SLOS.items()
    }
    out: dict[str, Any] = {"cost": args.cost, "slos": {k: str(v) for k, v in slos.items()}}

    # 1. policies --------------------------------------------------------------
    out["policies"] = {}
    for kind, slo in slos.items():
        base = azure(kind).clip(MAX_MODEL_LEN).head(args.requests)
        grid = (
            [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
            if kind == "conv"
            else [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
        )
        rates = [r * args.rate_scale for r in grid]
        rows = {}
        for label, (policy, cfg) in configs(args.num_blocks).items():
            t0 = time.perf_counter()
            sweep = rate_sweep(base, rates, cost, policy, cfg, eng, slo)
            cap = capacity(base, cost, slo, 0.9, policy, cfg, eng, lo=0.05, hi=64.0, rel_tol=0.005)
            # Peak throughput: every request arrives within a second (offline batch).
            peak = rate_sweep(base, [len(base)], cost, policy, cfg, eng)[0]
            rows[label] = {
                "sweep": sweep,
                "goodput_rps": cap["rate"],
                "at_goodput": cap["summary"],
                "peak_rps": peak["throughput_rps"],
                "peak_output_tok_s": peak["output_tok_s"],
                "wall_s": time.perf_counter() - t0,
            }
            print(
                f"{kind:>4} {label:>22}: goodput {cap['rate']:6.2f} req/s ({slo}); "
                f"peak {peak['throughput_rps']:6.2f} req/s"
            )
        out["policies"][kind] = rows

    # 2. token budget ----------------------------------------------------------
    out["budget"] = {}
    for kind, slo in slos.items():
        base = azure(kind).clip(MAX_MODEL_LEN).head(args.requests)
        rows = {}
        for budget in (512, 1024, 2048, 4096, 8192, 16384):
            cfg = SchedulerConfig(
                max_num_seqs=1024,
                max_num_batched_tokens=budget,
                num_blocks=args.num_blocks,
                max_model_len=MAX_MODEL_LEN,
            )
            cap = capacity(base, cost, slo, 0.9, "vllm", cfg, eng, lo=0.05, hi=64.0, rel_tol=0.005)
            rows[str(budget)] = {"goodput_rps": cap["rate"], "at_goodput": cap["summary"]}
            print(f"{kind:>4} budget {budget:>5}: goodput {cap['rate']:6.2f} req/s")
        out["budget"][kind] = rows

    # 3. speed -----------------------------------------------------------------
    full = azure("conv").clip(MAX_MODEL_LEN)
    cfg = SchedulerConfig(
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        num_blocks=args.num_blocks,
        max_model_len=MAX_MODEL_LEN,
    )
    t0 = time.perf_counter()
    res = simulate(full, cost, "vllm", cfg, EngineConfig(record_steps=True))
    wall = time.perf_counter() - t0
    out["speed"] = {
        "requests": len(full),
        "trace_seconds": full.duration,
        "iterations": res.meta["num_iterations"],
        "wall_s": wall,
        "iterations_per_s": res.meta["num_iterations"] / wall,
    }
    print(
        f"full conv trace: {len(full)} requests, {full.duration / 60:.0f} min of traffic, "
        f"{res.meta['num_iterations']} iterations in {wall:.1f}s"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
