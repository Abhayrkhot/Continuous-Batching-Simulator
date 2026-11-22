"""Print the README's result tables (markdown) from results/h200/*/{validation,experiments}.json.

python scripts/report.py > /tmp/tables.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("results/h200")
LABEL = {"qwen7b": "Qwen2.5-7B", "qwen32b": "Qwen2.5-32B"}
KEYS = (
    "ttft_p50",
    "ttft_p90",
    "ttft_p99",
    "tpot_p50",
    "tpot_p90",
    "tpot_p99",
    "e2e_p50",
    "e2e_p99",
)


def load(profile: str, name: str) -> dict[str, Any] | None:
    p = RESULTS / profile / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{x:+.1%}".replace("%", " %")


def accuracy(profiles: list[str]) -> None:
    errs: dict[str, list[float]] = {k: [] for k in KEYS}
    n_runs = n_req = 0
    for prof in profiles:
        rep = load(prof, "validation.json")
        if not rep:
            continue
        for cfg in rep["configs"].values():
            for row in cfg["runs"]:
                n_runs += 1
                n_req += row["requests"]
                for k in KEYS:
                    if k in row["metrics"]:
                        errs[k].append(abs(row["metrics"][k]["rel_err"]))
    print(f"Accuracy over {n_runs} replays ({n_req:,} requests):\n")
    print("| metric | median abs. error | 90th pct abs. error | within 10 % |")
    print("|---|---:|---:|---:|")
    for k in KEYS:
        e = np.array(errs[k])
        if len(e):
            name = k.replace("_p", " p").upper().replace("E2E", "E2E")
            print(
                f"| {name} | {np.median(e):.1%} | {np.percentile(e, 90):.1%} | "
                f"{(e <= 0.10).mean():.0%} |"
            )
    print()


def saturated(row: dict[str, Any], base: dict[str, float]) -> bool:
    """Past saturation: measured TTFT p50 more than 2x the lowest-rate run of its group."""
    return row["metrics"]["ttft_p50"]["real"] > 2 * base[row["run"].split("_")[0]]


def bands(profiles: list[str]) -> None:
    rows_by_band: dict[str, list[dict[str, Any]]] = {"below saturation": [], "saturated": []}
    for prof in profiles:
        rep = load(prof, "validation.json")
        if not rep:
            continue
        for cfg in rep["configs"].values():
            base: dict[str, float] = {}
            for row in sorted(cfg["runs"], key=lambda r: r["rate"]):
                base.setdefault(row["run"].split("_")[0], row["metrics"]["ttft_p50"]["real"])
            for row in cfg["runs"]:
                band = "saturated" if saturated(row, base) else "below saturation"
                rows_by_band[band].append(row)
    print(
        "Median absolute error by load (saturated = measured TTFT p50 above 2x its "
        "lowest-rate value):\n"
    )
    print("| band | replays | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for band, rows in rows_by_band.items():
        if not rows:
            continue
        cells = []
        for k in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50", "e2e_p99"):
            e = [abs(r["metrics"][k]["rel_err"]) for r in rows if k in r["metrics"]]
            cells.append(f"{np.median(e):.1%}")
        print(f"| {band} | {len(rows)} | " + " | ".join(cells) + " |")
    print()


def ablation(profiles: list[str]) -> None:
    errs: dict[str, dict[str, list[float]]] = {}
    for prof in profiles:
        rep = load(prof, "validation.json")
        if not rep:
            continue
        for cfg in rep["configs"].values():
            for row in cfg["runs"]:
                for variant, m in row.get("ablation", {}).items():
                    d = errs.setdefault(variant, {})
                    for k, v in m.items():
                        d.setdefault(k, []).append(abs(v))
    if not errs:
        return
    keys = ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99")
    print("Ablation, median absolute error over all replays:\n")
    print("| model | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |")
    print("|---|---:|---:|---:|---:|")
    for variant, d in errs.items():
        print(f"| {variant} | " + " | ".join(f"{np.median(d[k]):.1%}" for k in keys) + " |")
    print()


def ranking(profile: str) -> None:
    """For workloads measured under two configs: does the simulator order them correctly?"""
    rep = load(profile, "validation.json")
    if not rep or len(rep["configs"]) < 2:
        return
    names = list(rep["configs"])
    a, b = names[0], names[1]
    ra = {r["run"]: r for r in rep["configs"][a]["runs"]}
    rb = {r["run"]: r for r in rep["configs"][b]["runs"]}
    agree = total = 0
    rows = []
    for run in sorted(set(ra) & set(rb), key=lambda r: (r.split("_")[0], float(r.split("_r")[1]))):
        for k in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p99"):
            ma, mb = ra[run]["metrics"][k], rb[run]["metrics"][k]
            real_d = ma["real"] - mb["real"]
            sim_d = ma["sim"] - mb["sim"]
            if abs(real_d) < 0.03 * max(ma["real"], mb["real"]):
                continue  # measured difference under 3 %: no ranking to get right
            total += 1
            agree += (real_d > 0) == (sim_d > 0)
            rows.append((run, k, real_d / mb["real"], sim_d / mb["sim"]))
    if not total:
        return
    print(
        f"{LABEL[profile]}, `{a}` vs `{b}` on the same workloads: the simulator orders the two "
        f"configurations correctly on {agree}/{total} (workload, metric) pairs where they "
        "differ by more than 3 %.\n"
    )
    print(f"| workload | metric | measured {a} vs {b} | predicted |")
    print("|---|---|---:|---:|")
    for run, k, rd, sd in rows:
        print(f"| {run} | {k.replace('_', ' ')} | {rd:+.0%} | {sd:+.0%} |")
    print()


def per_run(profile: str) -> None:
    rep = load(profile, "validation.json")
    if not rep:
        return
    for name, cfg in rep["configs"].items():
        sc = cfg["scheduler"]
        print(
            f"**{LABEL[profile]}, `{name}`** "
            f"(max_num_batched_tokens={sc['max_num_batched_tokens']}, "
            f"max_num_seqs={sc['max_num_seqs']}, {sc['num_blocks']:,} KV blocks)\n"
        )
        print(
            "| trace @ req/s | requests | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | "
            "TPOT p99 (ms) | E2E p99 (s) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(cfg["runs"], key=lambda r: (r["run"].split("_")[0], r["rate"])):
            m = row["metrics"]

            def cell(k: str, scale: float, fmt: str, m: dict[str, Any] = m) -> str:
                r, s = m[k]["real"] * scale, m[k]["sim"] * scale
                return f"{fmt.format(r)} / {fmt.format(s)} ({m[k]['rel_err']:+.0%})"

            trace = row["run"].split("_")[0]
            print(
                f"| {trace} @ {row['rate']:.1f} | {row['requests']} | "
                f"{cell('ttft_p50', 1e3, '{:.0f}')} | "
                f"{cell('ttft_p99', 1e3, '{:.0f}')} | {cell('tpot_p50', 1e3, '{:.1f}')} | "
                f"{cell('tpot_p99', 1e3, '{:.1f}')} | {cell('e2e_p99', 1, '{:.2f}')} |"
            )
        print()


def step_model(profile: str) -> None:
    rep = load(profile, "validation.json")
    if not rep:
        return
    meta = rep["model"]["meta"]
    h = meta["holdout_step_error"]
    steps = []
    for cfg in rep["configs"].values():
        for row in cfg["runs"]:
            if row.get("step_model"):
                steps.append(row["step_model"])
    total = sum(s["steps"] for s in steps)
    med = (
        np.average([s["median_ape"] for s in steps], weights=[s["steps"] for s in steps])
        if steps
        else float("nan")
    )
    print(
        f"{LABEL[profile]}: fitted on {meta['calibration_steps']:,} calibration iterations; "
        f"held-out median error {h['median_ape']:.1%} (mean {h['mape']:.1%}); on {int(total):,} "
        f"iterations of the validation replays, median error {med:.1%}. Frontend latency "
        f"{meta['frontend_latency_s'] * 1e3:.1f} ms, idle launch "
        f"{meta['idle_launch_latency_s'] * 1e3:.1f} ms.\n"
    )


def experiments(profile: str) -> None:
    exp = load(profile, "experiments.json")
    if not exp:
        return
    print(
        f"**Goodput, {LABEL[profile]} (simulated)**: highest request rate at which 90 % of "
        "requests meet the SLO.\n"
    )
    kinds = list(exp["policies"])
    print("| policy | " + " | ".join(f"{k}: {exp['slos'][k]}" for k in kinds) + " |")
    print("|---|" + "---:|" * len(kinds))
    labels = list(next(iter(exp["policies"].values())))
    for lab in labels:
        cells = [f"{exp['policies'][k][lab]['goodput_rps']:.2f} req/s" for k in kinds]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    print()
    print("| policy | " + " | ".join(f"{k}: peak throughput" for k in kinds) + " |")
    print("|---|" + "---:|" * len(kinds))
    for lab in labels:
        cells = [
            f"{exp['policies'][k][lab]['peak_rps']:.1f} req/s "
            f"({exp['policies'][k][lab]['peak_output_tok_s']:,.0f} tok/s)"
            for k in kinds
        ]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    print()
    print("| max_num_batched_tokens | " + " | ".join(kinds) + " |")
    print("|---:|" + "---:|" * len(kinds))
    for b in exp["budget"][kinds[0]]:
        print(
            f"| {b} | "
            + " | ".join(f"{exp['budget'][k][b]['goodput_rps']:.2f}" for k in kinds)
            + " |"
        )
    sp = exp["speed"]
    print(
        f"\nSpeed: {sp['requests']:,} requests ({sp['trace_seconds'] / 60:.0f} min of traffic), "
        f"{sp['iterations']:,} iterations in {sp['wall_s']:.1f} s "
        f"({sp['iterations_per_s'] / 1e3:.0f}k iterations/s).\n"
    )


def main() -> None:
    profiles = [p for p in ("qwen7b", "qwen32b") if (RESULTS / p).exists()]
    accuracy(profiles)
    bands(profiles)
    ablation(profiles)
    for p in profiles:
        step_model(p)
    for p in profiles:
        ranking(p)
    for p in profiles:
        per_run(p)
    for p in profiles:
        experiments(p)
    parity = Path("results/parity_vllm034.json")
    if parity.exists():
        t = json.loads(parity.read_text())["totals"]
        print(
            f"Scheduler parity vs vLLM {t['vllm_version']}: {t['scenarios']} scenarios, "
            f"{t['steps']:,} steps, {t['mismatched_scenarios']} mismatches, "
            f"{t['preempting_scenarios']} scenarios with preemption."
        )


if __name__ == "__main__":
    main()
