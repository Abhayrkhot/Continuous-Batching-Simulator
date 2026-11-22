"""Render README.md from scripts/README.tmpl.md and the results (no hand-typed numbers).

python scripts/build_readme.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("report", HERE / "report.py")
assert spec and spec.loader
report: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def capture(fn: Any, *args: Any) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().strip()


def headline(profiles: list[str]) -> dict[str, str]:
    errs: dict[str, list[float]] = {}
    n_runs = n_req = 0
    for prof in profiles:
        rep = report.load(prof, "validation.json")
        for cfg in rep["configs"].values():
            for row in cfg["runs"]:
                n_runs += 1
                n_req += row["requests"]
                for k, m in row["metrics"].items():
                    errs.setdefault(k, []).append(abs(m["rel_err"]))

    def med(k: str) -> str:
        return f"{np.median(errs[k]):.1%}"

    return {
        "N_REPLAYS": str(n_runs),
        "N_REQUESTS": f"{n_req:,}",
        "E_TTFT50": med("ttft_p50"),
        "E_TTFT99": med("ttft_p99"),
        "E_TPOT50": med("tpot_p50"),
        "E_TPOT99": med("tpot_p99"),
        "E_E2E99": med("e2e_p99"),
    }


POLICIES = (
    ("static (batch 128)", "static (batch 128)"),
    ("orca (reserve 2048)", "orca (reserve 2048 tokens)"),
    ("orca (oracle)", "orca (oracle reservation)"),
    ("prefill-first", "prefill-first (vLLM V0)"),
    ("vllm v1 (budget 8192)", "vLLM V1, budget 8192"),
)


def policy_table() -> str:
    e7 = report.load("qwen7b", "experiments.json")
    e32 = report.load("qwen32b", "experiments.json")
    head = (
        "| policy | 7B conv goodput | 7B conv peak | 7B code peak | 32B conv goodput | "
        "32B conv peak | 32B code peak |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [head]
    for key, label in POLICIES:
        a, b = e7["policies"], e32["policies"]
        rows.append(
            f"| {label} | {a['conv'][key]['goodput_rps']:.1f} req/s | "
            f"{a['conv'][key]['peak_rps']:.1f} req/s | {a['code'][key]['peak_rps']:.1f} req/s | "
            f"{b['conv'][key]['goodput_rps']:.2f} req/s | {b['conv'][key]['peak_rps']:.1f} req/s | "
            f"{b['code'][key]['peak_rps']:.1f} req/s |"
        )
    return "\n".join(rows)


def main() -> None:
    profiles = [
        p for p in ("qwen7b", "qwen32b") if (report.RESULTS / p / "validation.json").exists()
    ]
    acc = capture(report.accuracy, profiles).split("\n", 2)[-1]  # drop the lead-in line
    ranking = capture(report.ranking, "qwen7b")
    m = re.search(r"correctly on (\d+/\d+)", ranking)
    values = {
        **headline(profiles),
        "ACCURACY_TABLE": acc,
        "BANDS_TABLE": capture(report.bands, profiles).split("\n", 2)[-1],
        "ABLATION_TABLE": capture(report.ablation, profiles).split("\n", 2)[-1],
        "RANKING_TABLE": ranking,
        "RANK_AGREE": (m.group(1).replace("/", " of ") if m else "?"),
        "STEP_MODEL_LINES": "\n\n".join(capture(report.step_model, p) for p in profiles),
        "PER_RUN_TABLES": "\n\n".join(capture(report.per_run, p) for p in profiles),
        "POLICY_TABLE": policy_table(),
    }
    text = (HERE / "README.tmpl.md").read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(\w+)\}\}", text)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (HERE.parent / "README.md").write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    main()
