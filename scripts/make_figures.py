"""Regenerate every figure in docs/ from results/ (no GPU needed).

python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

from batchsim.calibrate import fit_linear, periods
from batchsim.validation import RunDir

RESULTS = Path("results/h200")
DOCS = Path("docs")

# Validated light-mode categorical palette (dataviz reference instance), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9.5,
    }
)


def plain_log_axis(axis: Any) -> None:
    """Log scale, but ticks written as plain numbers (20, 50, 100) instead of 2x10^1."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def runs_by_trace(cfg: dict[str, Any], trace: str) -> list[dict[str, Any]]:
    rows = [r for r in cfg["runs"] if r["run"].startswith(trace)]
    return sorted(rows, key=lambda r: r["rate"])


def parity() -> None:
    """Predicted vs measured latency percentiles, every run, every config."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    groups = {"TTFT": ("ttft", BLUE), "TPOT": ("tpot", ORANGE), "End-to-end": ("e2e", AQUA)}
    allv = []
    for profile, marker in (("qwen7b", "o"), ("qwen32b", "s")):
        rep = load(f"{profile}/validation.json")
        if not rep:
            continue
        for cfg in rep["configs"].values():
            for row in cfg["runs"]:
                for key, color in groups.values():
                    for p in (50, 90, 99):
                        m = row["metrics"].get(f"{key}_p{p}")
                        if not m:
                            continue
                        ax.scatter(
                            m["real"],
                            m["sim"],
                            s=16,
                            color=color,
                            marker=marker,
                            alpha=0.75,
                            linewidths=0,
                        )
                        allv += [m["real"], m["sim"]]
    if not allv:
        plt.close(fig)
        return
    lo, hi = min(allv) / 1.5, max(allv) * 1.5
    xs = np.geomspace(lo, hi, 50)
    ax.plot(xs, xs, color=INK2, lw=1)
    ax.fill_between(xs, xs / 1.1, xs * 1.1, color="#d6d4cd", alpha=0.7, lw=0, label="±10%")
    for label, (_, color) in groups.items():
        ax.scatter([], [], color=color, s=20, label=label)
    ax.scatter([], [], color=INK2, marker="o", s=20, label="Qwen2.5-7B")
    ax.scatter([], [], color=INK2, marker="s", s=20, label="Qwen2.5-32B")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel("measured on vLLM 0.34, H200 (s)")
    ax.set_ylabel("predicted by batchsim (s)")
    ax.set_title("p50 / p90 / p99 latency, every replay")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "parity.png", dpi=160)
    plt.close(fig)


def latency_vs_rate(profile: str = "qwen7b", trace: str = "conv") -> None:
    rep = load(f"{profile}/validation.json")
    if not rep:
        return
    cfgs = [(name, c) for name, c in rep["configs"].items() if runs_by_trace(c, trace)]
    if not cfgs:
        return
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    panels = [
        ("ttft_p50", "TTFT p50"),
        ("ttft_p99", "TTFT p99"),
        ("tpot_p50", "TPOT p50"),
        ("tpot_p99", "TPOT p99"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for (_name, cfg), color in zip(cfgs, colors):
            rows = runs_by_trace(cfg, trace)
            rate = [r["rate"] for r in rows]
            real = [r["metrics"][key]["real"] * 1e3 for r in rows]
            sim = [r["metrics"][key]["sim"] * 1e3 for r in rows]
            budget = cfg["scheduler"]["max_num_batched_tokens"]
            ax.plot(rate, real, color=color, marker="o", label=f"budget {budget}: measured")
            ax.plot(rate, sim, color=color, ls="--", label=f"budget {budget}: batchsim")
        ax.set_title(title)
        ax.set_ylabel("ms")
        if key.startswith("ttft"):
            ax.set_yscale("log")
            plain_log_axis(ax.yaxis)
    for ax in axes[1]:
        ax.set_xlabel("request rate (req/s)")
    axes[0, 0].legend(loc="upper left")
    name = {"conv": "conversation", "code": "code"}[trace]
    fig.suptitle(
        f"{profile_label(profile)} on one H200, Azure {name} trace", color=INK, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(DOCS / f"{trace}_{profile}.png", dpi=160)
    plt.close(fig)


def profile_label(profile: str) -> str:
    return {"qwen7b": "Qwen2.5-7B", "qwen32b": "Qwen2.5-32B"}[profile]


def step_model(profile: str = "qwen7b") -> None:
    """Held-out iterations: predicted vs measured period."""
    root = RESULTS / profile
    if not root.exists():
        return
    cal = next(
        (
            RunDir.load(d)
            for d in sorted(root.iterdir())
            if (d / "runs.jsonl").exists() and "calibration" in RunDir.load(d).windows
        ),
        None,
    )
    if cal is None:
        return
    steps = cal.steps_in("calibration")
    y = periods(steps)
    rng = np.random.default_rng(0)
    test = rng.random(len(y)) < 0.2
    model, _ = fit_linear([s.stats for s, t in zip(steps, test) if not t], y[~test])
    yt = y[test] * 1e3
    pt = np.array([model.step_time(s.stats) for s, t in zip(steps, test) if t]) * 1e3
    mixed = np.array([s.stats.num_prefill > 0 for s, t in zip(steps, test) if t])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(
        yt[~mixed], pt[~mixed], s=6, color=BLUE, alpha=0.4, linewidths=0, label="decode only"
    )
    ax.scatter(
        yt[mixed], pt[mixed], s=8, color=ORANGE, alpha=0.6, linewidths=0, label="with prefill"
    )
    lo, hi = min(yt.min(), pt.min()) / 1.2, max(yt.max(), pt.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
    ape = np.abs(pt - yt) / yt
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel("measured iteration time (ms)")
    ax.set_ylabel("predicted (ms)")
    ax.set_title(
        f"{profile_label(profile)} step model, {test.sum()} held-out iterations\n"
        f"median error {np.median(ape):.1%}, mean {ape.mean():.1%}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / f"step_model_{profile}.png", dpi=160)
    plt.close(fig)


POLICY_STYLE = {
    "static (batch 128)": YELLOW,
    "orca (reserve 2048)": AQUA,
    "prefill-first": ORANGE,
    "vllm v1 (budget 8192)": BLUE,
}


def policies(profile: str = "qwen7b") -> None:
    """Left: SLO attainment vs rate on the conversation trace. Right: peak throughput."""
    exp = load(f"{profile}/experiments.json")
    if not exp:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = exp["policies"]["conv"]
    for label, color in POLICY_STYLE.items():
        if label not in rows:
            continue
        sw = rows[label]["sweep"]
        ax.plot(
            [r["rate"] for r in sw],
            [100 * r["slo_attainment"] for r in sw],
            color=color,
            marker="o",
            label=f"{label}: {rows[label]['goodput_rps']:.1f} req/s",
        )
    ax.axhline(90, color=INK2, lw=1, ls=":")
    ax.set_ylim(0, 104)
    ax.set_xlabel("request rate (req/s)")
    ax.set_ylabel("requests meeting SLO (%)")
    ax.set_title(f"Azure conv, SLO {exp['slos']['conv']}", fontsize=10)
    ax.legend(loc="center left", title="goodput (90% meet SLO)", title_fontsize=8.5)

    labels = [lab for lab in POLICY_STYLE if lab in rows]
    y = np.arange(len(labels))
    h = 0.36
    for k, (kind, color, off) in enumerate((("conv", BLUE, -h / 2), ("code", ORANGE, h / 2))):
        vals = [exp["policies"][kind][lab]["peak_rps"] for lab in labels]
        bx.barh(y + off, vals, height=h - 0.04, color=color, label=f"Azure {kind}")
        for yi, v in zip(y + off, vals):
            bx.text(v, yi, f" {v:.1f}", va="center", ha="left", fontsize=8.5, color=INK2)
        del k
    bx.set_yticks(y, labels)
    bx.invert_yaxis()
    bx.set_xlabel("peak throughput (req/s), all requests queued at once")
    bx.set_title("Peak throughput", fontsize=10)
    bx.grid(axis="y", visible=False)
    bx.set_xlim(
        0,
        max(exp["policies"][k_][lab]["peak_rps"] for k_ in ("conv", "code") for lab in labels)
        * 1.18,
    )
    bx.legend(loc="upper right")
    fig.suptitle(
        f"Batching policies, {profile_label(profile)} on one H200 (simulated)",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(DOCS / f"policies_{profile}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    parity()
    for profile in ("qwen7b", "qwen32b"):
        for trace in ("conv", "code"):
            latency_vs_rate(profile, trace)
        step_model(profile)
        policies(profile)
    print("figures:", sorted(p.name for p in DOCS.glob("*.png")))


if __name__ == "__main__":
    main()
