"""Command-line interface: ``batchsim {simulate,compare,sweep,capacity,calibrate,workload}``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from batchsim.calibrate import calibrate_from_probe
from batchsim.cost import RooflineModel, StepTimeModel, load_cost_model
from batchsim.engine import EngineConfig, simulate
from batchsim.hardware import get_gpu, get_model, kv_cache_blocks
from batchsim.metrics import SLO
from batchsim.schedulers import POLICIES, SchedulerConfig
from batchsim.sweep import capacity, rate_sweep
from batchsim.workload import LogNormal, Workload, azure, synthetic

DEFAULT_COLUMNS = (
    ("rate", "rate", 1.0, "{:.2f}"),
    ("tput r/s", "throughput_rps", 1.0, "{:.2f}"),
    ("out tok/s", "output_tok_s", 1.0, "{:.0f}"),
    ("TTFT p50 ms", "ttft_p50", 1e3, "{:.0f}"),
    ("TTFT p99 ms", "ttft_p99", 1e3, "{:.0f}"),
    ("TPOT p50 ms", "tpot_p50", 1e3, "{:.1f}"),
    ("TPOT p99 ms", "tpot_p99", 1e3, "{:.1f}"),
    ("E2E p50 s", "e2e_p50", 1.0, "{:.2f}"),
    ("batch", "mean_batch_seqs", 1.0, "{:.0f}"),
    ("preempt", "preemptions", 1.0, "{:.0f}"),
)


def packaged_cost_models() -> list[str]:
    root = resources.files("batchsim") / "data"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".json"))


def resolve_cost_model(args: argparse.Namespace) -> StepTimeModel:
    if args.cost:
        path = Path(args.cost)
        if path.exists():
            return load_cost_model(path)
        packaged = resources.files("batchsim") / "data" / f"{args.cost}.json"
        if packaged.is_file():
            with resources.as_file(packaged) as p:
                return load_cost_model(p)
        raise SystemExit(f"no cost model {args.cost!r}; packaged: {packaged_cost_models()}")
    return RooflineModel(get_model(args.model), get_gpu(args.gpu), args.tensor_parallel)


def load_workload(args: argparse.Namespace) -> Workload:
    spec = args.workload
    if spec in ("azure-conv", "azure-code"):
        wl = azure(spec.split("-", 1)[1], args.data_dir)
        if args.n:
            wl = wl.head(args.n)
        if args.rate:
            wl = wl.scale_to_rate(args.rate)
    elif spec == "synthetic":
        wl = synthetic(
            args.n or 2000,
            args.rate or 10.0,
            LogNormal(args.prompt_mean, args.length_cv, 1, args.max_model_len // 2),
            LogNormal(args.output_mean, args.length_cv, 1, args.max_model_len // 2),
            process="gamma" if args.arrival_cv != 1.0 else "poisson",
            cv=args.arrival_cv,
            seed=args.seed,
        )
    else:
        wl = Workload.from_csv(spec)
        if args.n:
            wl = wl.head(args.n)
        if args.rate:
            wl = wl.scale_to_rate(args.rate)
    return wl.clip(args.max_model_len)


def scheduler_config(args: argparse.Namespace, policy: str) -> SchedulerConfig:
    num_blocks = args.num_blocks or kv_cache_blocks(
        get_model(args.model),
        get_gpu(args.gpu),
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        tensor_parallel=args.tensor_parallel,
    )
    budget = args.max_num_batched_tokens
    if policy == "prefill-first":
        budget = max(budget, args.max_model_len)
    elif policy == "orca":
        budget = 1 << 30
    return SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=budget,
        num_blocks=num_blocks,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=not args.no_chunked_prefill,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        watermark=0.01 if policy == "prefill-first" else args.watermark,
        reserve_max_tokens=args.reserve_max_tokens,
        max_wait=args.max_wait,
    )


def engine_config(args: argparse.Namespace, cost: StepTimeModel | None = None) -> EngineConfig:
    """Engine settings; frontend latencies default to the ones calibrated with ``cost``."""
    meta = getattr(cost, "meta", {}) or {}
    frontend = args.frontend_latency
    if frontend is None:
        frontend = float(meta.get("frontend_latency_s", 0.0))
    idle = args.idle_launch_latency
    if idle is None:
        idle = float(meta.get("idle_launch_latency_s", 0.0))
    return EngineConfig(
        async_scheduling=not args.no_async_scheduling,
        frontend_latency=frontend,
        idle_launch_latency=idle,
    )


def slo_from(args: argparse.Namespace) -> SLO | None:
    if args.slo_ttft is None and args.slo_tpot is None and args.slo_e2e is None:
        return None
    return SLO(
        ttft=args.slo_ttft / 1e3 if args.slo_ttft is not None else math.inf,
        tpot=args.slo_tpot / 1e3 if args.slo_tpot is not None else math.inf,
        e2e=args.slo_e2e if args.slo_e2e is not None else math.inf,
    )


def print_table(rows: Sequence[dict[str, Any]], label: str, extra: Sequence[str] = ()) -> None:
    cols = [(label, label, 1.0, "{}"), *DEFAULT_COLUMNS]
    if any("slo_attainment" in r for r in rows):
        cols.append(("SLO %", "slo_attainment", 100.0, "{:.1f}"))
    cols.extend((e, e, 1.0, "{}") for e in extra)
    cells = [[c[0] for c in cols]]
    for r in rows:
        line = []
        for _, key, scale, fmt in cols:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                line.append("-")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                line.append(fmt.format(v * scale))
            else:
                line.append(str(v))
        cells.append(line)
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    for k, row in enumerate(cells):
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
        if k == 0:
            print("  ".join("-" * w for w in widths))


def window_frac(args: argparse.Namespace) -> tuple[float, float]:
    return args.warmup, args.cooldown


# ---------------------------------------------------------------------- commands


def cmd_simulate(args: argparse.Namespace) -> int:
    wl = load_workload(args)
    cost = resolve_cost_model(args)
    res = simulate(
        wl, cost, args.policy, scheduler_config(args, args.policy), engine_config(args, cost)
    )
    w, c = window_frac(args)
    s = res.summary(wl.duration * w, wl.duration * c, slo_from(args))
    s["rate"] = wl.rate
    s["sim_wall_s"] = res.meta["wall_s"]
    print_table([{args.policy: args.policy, **s}], args.policy)
    print(
        f"\n{len(wl)} requests, {res.meta['num_iterations']} iterations simulated in "
        f"{res.meta['wall_s']:.2f}s ({cost.kind} step model)"
    )
    if args.json:
        res.to_json(args.json, summary_window=s)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    wl = load_workload(args)
    cost = resolve_cost_model(args)
    rows = []
    w, c = window_frac(args)
    for pol in args.policies:
        res = simulate(wl, cost, pol, scheduler_config(args, pol), engine_config(args, cost))
        s = res.summary(wl.duration * w, wl.duration * c, slo_from(args))
        rows.append({"policy": pol, "rate": wl.rate, **s})
    print_table(rows, "policy")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    args.rate = None
    base = load_workload(args)
    cost = resolve_cost_model(args)
    rows = rate_sweep(
        base,
        args.rates,
        cost,
        args.policy,
        scheduler_config(args, args.policy),
        engine_config(args, cost),
        slo_from(args),
        args.workers,
    )
    print_table([{"policy": args.policy, **r} for r in rows], "policy")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    slo = slo_from(args)
    if slo is None:
        raise SystemExit("capacity needs at least one of --slo-ttft/--slo-tpot/--slo-e2e")
    args.rate = None
    base = load_workload(args)
    cost = resolve_cost_model(args)
    rows = []
    for pol in args.policies:
        budgets = args.budgets if pol == "vllm" and args.budgets else [args.max_num_batched_tokens]
        for b in budgets:
            args.max_num_batched_tokens = b
            cap = capacity(
                base,
                cost,
                slo,
                args.attainment,
                pol,
                scheduler_config(args, pol),
                engine_config(args, cost),
                hi=args.max_rate,
            )
            label = pol if len(budgets) == 1 else f"{pol} budget={b}"
            rows.append({"config": label, **cap["summary"], "rate": cap["rate"]})
    print(f"Highest rate with >= {args.attainment:.0%} of requests meeting {slo}:\n")
    print_table(rows, "config")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    model, report = calibrate_from_probe(
        args.steps, idle_gap=args.idle_gap, holdout=args.holdout, meta={"source": args.label}
    )
    model.save(args.out)
    h = report["holdout"]
    print(
        f"fitted on {report['steps']} iterations; held-out error: mean {h['mape']:.1%}, "
        f"median {h['median_ape']:.1%}, p90 {h['p90_ape']:.1%}, bias {h['bias']:+.1%}"
    )
    for k, v in model.coef.items():
        print(f"  {k:>16} = {v:.4g}")
    print(f"wrote {args.out}")
    return 0


def cmd_workload(args: argparse.Namespace) -> int:
    wl = load_workload(args)
    wl.to_csv(args.out)
    print(f"wrote {len(wl)} requests ({wl.rate:.2f} req/s over {wl.duration:.0f}s) to {args.out}")
    return 0


# ---------------------------------------------------------------------- parser


def _add_workload_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("workload")
    g.add_argument(
        "--workload",
        default="azure-conv",
        help="azure-conv, azure-code, synthetic, or a CSV file (arrival,prompt_len,output_len)",
    )
    g.add_argument("--n", type=int, default=0, help="use the first N requests")
    g.add_argument("--rate", type=float, default=None, help="time-scale to this mean req/s")
    g.add_argument("--data-dir", default="data", help="where Azure traces are cached")
    g.add_argument("--prompt-mean", type=float, default=1000.0)
    g.add_argument("--output-mean", type=float, default=200.0)
    g.add_argument("--length-cv", type=float, default=1.0)
    g.add_argument("--arrival-cv", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-model-len", type=int, default=16384)


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("engine")
    g.add_argument("--cost", help="cost model JSON, or a packaged name (e.g. h200-qwen2.5-7b)")
    g.add_argument("--model", default="qwen2.5-7b", help="model for roofline/KV sizing")
    g.add_argument("--gpu", default="h200")
    g.add_argument("--tensor-parallel", type=int, default=1)
    g.add_argument("--num-blocks", type=int, default=0, help="KV blocks (default: from GPU)")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    g.add_argument("--block-size", type=int, default=16)
    g.add_argument("--max-num-seqs", type=int, default=1024)
    g.add_argument("--max-num-batched-tokens", type=int, default=8192)
    g.add_argument("--no-chunked-prefill", action="store_true")
    g.add_argument("--long-prefill-token-threshold", type=int, default=0)
    g.add_argument("--watermark", type=float, default=0.0)
    g.add_argument("--reserve-max-tokens", type=int, default=None)
    g.add_argument("--max-wait", type=float, default=0.0, help="static batching timeout (s)")
    g.add_argument("--no-async-scheduling", action="store_true")
    g.add_argument(
        "--frontend-latency",
        type=float,
        default=None,
        help="seconds added to TTFT/E2E (default: the calibrated value, else 0)",
    )
    g.add_argument(
        "--idle-launch-latency",
        type=float,
        default=None,
        help="seconds before the first step after idle (default: calibrated, else 0)",
    )
    s = p.add_argument_group("SLO / reporting")
    s.add_argument("--slo-ttft", type=float, help="ms")
    s.add_argument("--slo-tpot", type=float, help="ms")
    s.add_argument("--slo-e2e", type=float, help="s")
    s.add_argument("--warmup", type=float, default=0.1, help="fraction of the span to skip")
    s.add_argument("--cooldown", type=float, default=0.1, help="fraction of the span to skip")
    s.add_argument("--json", help="write results as JSON")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="batchsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="simulate one policy on one workload")
    _add_workload_args(p)
    _add_engine_args(p)
    p.add_argument("--policy", choices=POLICIES, default="vllm")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("compare", help="compare policies on the same workload")
    _add_workload_args(p)
    _add_engine_args(p)
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="latency and throughput across request rates")
    _add_workload_args(p)
    _add_engine_args(p)
    p.add_argument("--policy", choices=POLICIES, default="vllm")
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("capacity", help="highest rate that meets an SLO (goodput)")
    _add_workload_args(p)
    _add_engine_args(p)
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=["vllm"])
    p.add_argument("--budgets", type=int, nargs="*", help="vllm token budgets to search")
    p.add_argument("--attainment", type=float, default=0.9)
    p.add_argument("--max-rate", type=float, default=200.0)
    p.set_defaults(func=cmd_capacity)

    p = sub.add_parser("calibrate", help="fit a step-time model to vLLM probe logs")
    p.add_argument("steps", nargs="+", help="steps.jsonl files from bench/batchsim_vllm_probe")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--idle-gap", type=float, default=2e-3)
    p.add_argument("--holdout", type=float, default=0.2)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("workload", help="export a workload as CSV")
    _add_workload_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_workload)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
