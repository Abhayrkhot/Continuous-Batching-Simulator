#!/bin/bash
# Serve a model with vLLM 0.34 on one H200 and replay workloads against it, recording
# per-request token times (client) and per-iteration batch shapes and GPU times (probe).
#
#   PROFILE=qwen7b CONFIG=default RUNS="calibration conv_r8" sbatch bench/explorer.sbatch
#   PROFILE=qwen7b CONFIG=b2048 VLLM_ARGS="--max-num-batched-tokens 2048 --max-num-seqs 256" \
#       sbatch bench/explorer.sbatch
#
# Runs inside the vLLM Docker image converted to Apptainer (see README, "Reproducing").
# Explorer's driver (570, CUDA 12.8) is older than the image's CUDA 13, so the image's
# forward-compatibility libraries are put on LD_LIBRARY_PATH.
#
#SBATCH --job-name=cbsim
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:59:00
#SBATCH --output=%x-%j.log

set -uo pipefail
PROFILE=${PROFILE:?set PROFILE=qwen7b|qwen32b}
CONFIG=${CONFIG:-default}
VLLM_ARGS=${VLLM_ARGS:-}
ROOT=${CBSIM_ROOT:-/scratch/$USER/cbsim}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
REPO=$ROOT/repo
WL=$REPO/bench/workloads/$PROFILE
case $PROFILE in
  qwen7b) MODEL=Qwen/Qwen2.5-7B-Instruct ;;
  qwen32b) MODEL=Qwen/Qwen2.5-32B-Instruct ;;
  *) echo "unknown PROFILE $PROFILE"; exit 2 ;;
esac
if [ -z "${RUNS:-}" ]; then
  RUNS=$(python3 -c "import json,sys; print(' '.join(m['name'] for m in json.load(open('$WL/manifest.json'))))")
fi
OUT=$ROOT/results/$PROFILE/${CONFIG}-${SLURM_JOB_ID:-local}
mkdir -p "$OUT" "$ROOT/cache"

export HF_HOME=${HF_HOME:-/scratch/$USER/hf} HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
bash "$REPO/bench/install_probe.sh" "$ROOT/probe" >/dev/null
export PYTHONPATH=$ROOT/probe
export BATCHSIM_STEP_LOG=$OUT/steps.jsonl
module load apptainer 2>/dev/null || true

in_image() {
  apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
    export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    exec "$@"' _ "$@"
}

nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$OUT/gpu.txt"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
URL=http://127.0.0.1:$PORT
echo "serving $MODEL ($CONFIG: $VLLM_ARGS) on $URL; runs: $RUNS"
SERVER=""
start_server() {  # (re)start vLLM and wait until it answers; logs append across restarts
  # shellcheck disable=SC2086  # VLLM_ARGS is intentionally word-split
  in_image vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name m \
    --max-model-len 16384 --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    $VLLM_ARGS >>"$OUT/server.log" 2>&1 &
  SERVER=$!
  local t0; t0=$(date +%s)
  until curl -sf "$URL/health" >/dev/null; do
    if ! kill -0 "$SERVER" 2>/dev/null; then echo "server died during startup"; tail -50 "$OUT/server.log"; return 1; fi
    if (( $(date +%s) - t0 > 1200 )); then echo "server not ready after 20 min"; return 1; fi
    sleep 5
  done
  echo "server ready after $(( $(date +%s) - t0 ))s (pid $SERVER)"
}
stop_server() { [ -n "$SERVER" ] && { kill -INT "$SERVER" 2>/dev/null; sleep 10; kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; }; }
trap 'stop_server; kill $MEMLOG 2>/dev/null' EXIT

# Every 15 s: node memory (used/available MiB), our RSS, and the job cgroup's memory and
# OOM-kill counters, so an unexplained server kill can be attributed.
CG=/sys/fs/cgroup$(awk -F: '$2 == "" || $2 ~ /memory/ {print $3; exit}' /proc/self/cgroup)
( while true; do
    oom=$(awk '/oom_kill/ {print $2}' "$CG/memory.events" 2>/dev/null || cat "$CG/memory.failcnt" 2>/dev/null)
    cur=$(cat "$CG/memory.current" 2>/dev/null || cat "$CG/memory.usage_in_bytes" 2>/dev/null)
    echo "$(date +%s) $(free -m | awk '/Mem:/ {print $3, $7}') $(ps -u "$USER" -o rss= | awk '{s+=$1} END {print int(s/1024)}') cg_mb=$(( ${cur:-0} / 1048576 )) oom=${oom:-?}"
    sleep 15
  done ) >"$OUT/mem.log" 2>&1 &
MEMLOG=$!

start_server || exit 1

metric() { curl -s "$URL/metrics" | awk -v m="$1" '$1 ~ "^"m"\\{" {s+=$2} END {print s+0}'; }
wait_idle() {
  for _ in $(seq 1 600); do
    r=$(metric vllm:num_requests_running); w=$(metric vllm:num_requests_waiting)
    if [ "${r%.*}" = 0 ] && [ "${w%.*}" = 0 ]; then return 0; fi
    sleep 1
  done
  echo "engine not idle after 10 min"
}

cat >"$OUT/meta.json" <<EOF
{"profile": "$PROFILE", "config": "$CONFIG", "model": "$MODEL", "vllm_args": "$VLLM_ARGS",
 "job": "${SLURM_JOB_ID:-local}", "node": "$(hostname)", "sif": "$SIF"}
EOF

# Smoke check: a few requests must come back complete and the probe must be logging.
in_image python3 "$REPO/bench/replay.py" --url "$URL" --model m --workload "$WL/smoke.csv" \
  --out "$OUT/smoke.jsonl"
wait_idle; sleep 3
if ! python3 - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
recs = [json.loads(l) for l in (out / "smoke.jsonl").open() if '"summary"' not in l]
bad = [r for r in recs if r.get("error") or r.get("completion_tokens") != r["output_len"]
       or len(r.get("tok", ())) != r["output_len"]]
steps = sum(1 for _ in (out / "steps.jsonl").open()) if (out / "steps.jsonl").exists() else 0
print(f"smoke: {len(recs)} requests, {len(bad)} bad, {steps} probe steps")
if bad: print("first bad:", {k: v for k, v in bad[0].items() if k != "tok"})
sys.exit(0 if recs and not bad and steps > 0 else 1)
PY
then echo "smoke check failed"; tail -40 "$OUT/server.log"; exit 1; fi

replay_ok() {  # every request of the run came back complete
  python3 - "$1" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if '"summary"' not in l]
bad = sum(1 for r in recs if r.get("error") or len(r.get("tok", ())) != r["output_len"])
print(f"  {len(recs) - bad}/{len(recs)} requests complete")
sys.exit(1 if bad or not recs else 0)
PY
}

for run in $RUNS; do
  for attempt in 1 2; do
    echo "== $run (attempt $attempt)"
    if ! curl -sf "$URL/health" >/dev/null; then
      echo "server is down; restarting"; stop_server; start_server || exit 1
    fi
    start=$(python3 -c 'import time; print(time.time())')
    in_image python3 "$REPO/bench/replay.py" --url "$URL" --model m \
      --workload "$WL/$run.csv" --out "$OUT/$run.jsonl" || echo "replay $run failed"
    wait_idle
    end=$(python3 -c 'import time; print(time.time())')
    if replay_ok "$OUT/$run.jsonl"; then
      echo "{\"run\": \"$run\", \"start\": $start, \"end\": $end}" >>"$OUT/runs.jsonl"
      break
    fi
    mv "$OUT/$run.jsonl" "$OUT/$run.failed$attempt.jsonl"
  done
  sleep 3
done
curl -s "$URL/metrics" >"$OUT/metrics_final.txt"
echo "done: $OUT"
