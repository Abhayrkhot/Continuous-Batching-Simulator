"""Generate the calibration and validation workloads replayed against vLLM on the GPU.

    python scripts/make_workloads.py --profile qwen7b --out bench/workloads/qwen7b

The calibration workload is synthetic and covers the batch shapes the engine will see
(isolated prefills, decode ladders, mixed Poisson traffic, a saturated prefill burst); the
step-time model is fitted on it alone. Validation workloads are windows of the Azure LLM
inference traces (conversation and code), time-scaled to a ladder of request rates. The
simulator never sees validation data before predicting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from batchsim.request import RequestSpec
from batchsim.workload import LogNormal, Workload, azure, synthetic

MAX_MODEL_LEN = 16384
RUN_SECONDS = 120.0
MAX_RUN_SECONDS = 240.0

PROFILES = {
    # rates in req/s; decode ladder (batch, prompt) pairs; poisson mixture rates
    "qwen7b": {
        "conv": [4, 8, 12, 16, 20, 24, 28],
        "code": [1, 2, 3, 4, 6],
        "ladder": [(1, 512), (8, 512), (32, 1024), (64, 4096), (128, 1024), (256, 2048)],
        "mix": [6, 12, 20, 28],
        "burst": 300,
    },
    "qwen32b": {
        "conv": [1, 2, 3, 4, 5, 6],
        "code": [0.5, 1, 1.5, 2],
        "ladder": [(1, 512), (8, 512), (32, 1024), (64, 2048), (128, 1024)],
        "mix": [1.5, 3, 4.5, 6],
        "burst": 120,
    },
}


def calibration(profile: dict[str, object], seed: int = 1234) -> Workload:
    rng = np.random.default_rng(seed)
    specs: list[RequestSpec] = []
    t = 0.0

    # 1. Isolated prefills of many lengths (also measures frontend latency at zero load).
    for p in rng.integers(16, 12000, size=40):
        specs.append(RequestSpec(t, int(p), 16))
        t += 1.5
    t += 5.0

    # 2. Decode ladder: B identical requests at once, 192 tokens each.
    for b, p in profile["ladder"]:  # type: ignore[attr-defined]
        for _ in range(b):
            specs.append(RequestSpec(t, int(p), 192))
        t += 12.0 if b < 128 else 20.0

    # 3. Poisson mixtures with heavy-tailed lengths.
    for rate in profile["mix"]:  # type: ignore[attr-defined]
        wl = synthetic(
            int(rate * 40),
            float(rate),
            LogNormal(1500, 1.0, 16, 12000),
            LogNormal(250, 1.0, 1, 1500),
            seed=int(rng.integers(1 << 30)),
        )
        specs.extend(RequestSpec(t + s.arrival, s.prompt_len, s.output_len) for s in wl.specs)
        t += wl.specs[-1].arrival + 15.0

    # 4. One saturated prefill burst.
    for p in LogNormal(2000, 0.8, 16, 12000).sample(rng, int(profile["burst"])):  # type: ignore[call-overload]
        specs.append(RequestSpec(t, int(p), 16))
    specs.sort(key=lambda s: s.arrival)
    return Workload(tuple(specs), "calibration").clip(MAX_MODEL_LEN)


def validation(kind: str, rate: float, data_dir: str) -> Workload:
    base = azure(kind, data_dir).clip(MAX_MODEL_LEN)
    # About RUN_SECONDS of traffic, at least 300 requests unless that exceeds MAX_RUN_SECONDS.
    n = int(min(1500, max(rate * RUN_SECONDS, min(300, rate * MAX_RUN_SECONDS))))
    wl = base.head(n).scale_to_rate(rate)
    return Workload(wl.specs, f"{kind}_r{rate:g}", {"trace": kind, "rate": rate})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    smoke = [RequestSpec(0.5 * i, 64 + 300 * i, 8 + i) for i in range(12)]
    smoke += [RequestSpec(7.0, 128 + 50 * i, 32) for i in range(24)]
    Workload(tuple(smoke), "smoke").to_csv(args.out / "smoke.csv")
    cal = calibration(prof)
    cal.to_csv(args.out / "calibration.csv")
    manifest.append({"name": "calibration", "requests": len(cal), "seconds": cal.duration})
    for kind in ("conv", "code"):
        for rate in prof[kind]:  # type: ignore[attr-defined]
            wl = validation(kind, float(rate), args.data)
            wl.to_csv(args.out / f"{wl.name}.csv")
            manifest.append({"name": wl.name, "requests": len(wl), "seconds": wl.duration})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(m["seconds"] for m in manifest)
    for m in manifest:
        print(f"{m['name']:>14}: {m['requests']:5d} requests over {m['seconds']:6.1f}s")
    print(f"total arrival span {total / 60:.1f} min")


if __name__ == "__main__":
    main()
