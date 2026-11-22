"""Compact raw run directories for the repository.

    python scripts/compact_results.py results/h200/qwen7b/default-12060102 [...]

Replay logs keep, per request, the send time, lengths, first/last token times and token
count (all the analysis uses); per-token times stay with the raw logs on the cluster. Probe
and server logs are gzipped unchanged. Already-compacted files are left alone.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

# Cluster paths carry the account name; keep logs shareable.
USER_PATH = re.compile(r"/(scratch|home|work)/[^/\s\"']+")
PROBE_AND_LOGS = ("steps.jsonl", "server.log")


def scrub(text: str) -> str:
    return USER_PATH.sub(lambda m: f"/{m.group(1)}/$USER", text)


def compact_replay(src: Path) -> None:
    dst = src.with_name(src.name + ".gz")
    with src.open() as f, gzip.open(dst, "wt") as out:
        for line in f:
            rec = json.loads(line)
            if "tok" in rec:
                toks = rec.pop("tok")
                rec["ntok"] = len(toks)
                if toks:
                    rec["first"], rec["last"] = toks[0], toks[-1]
            out.write(scrub(json.dumps(rec, separators=(",", ":"))) + "\n")
    src.unlink()


def main() -> None:
    for d in map(Path, sys.argv[1:]):
        for f in sorted(d.iterdir()):
            if f.suffix == ".gz" and f.is_file():  # already compacted: scrub in place
                with gzip.open(f, "rt", errors="replace") as i:
                    text = i.read()
                if USER_PATH.search(text):
                    with gzip.open(f, "wt") as o:
                        o.write(scrub(text))
                continue
            if not f.is_file():
                continue
            if f.name in PROBE_AND_LOGS:
                with gzip.open(f.with_name(f.name + ".gz"), "wt") as o:
                    o.write(scrub(f.read_text(errors="replace")))
                f.unlink()
            elif f.suffix in (".json", ".log", ".txt") or f.name == "runs.jsonl":
                f.write_text(scrub(f.read_text(errors="replace")))
            elif f.suffix == ".jsonl" and f.name != "runs.jsonl":
                compact_replay(f)
        size = sum(p.stat().st_size for p in d.iterdir())
        print(f"{d}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
