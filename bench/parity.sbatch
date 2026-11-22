#!/bin/bash
# Scheduler parity against vLLM's own scheduler, CPU only (see bench/scheduler_parity.py).
#   SCENARIOS=1000 sbatch bench/parity.sbatch
#SBATCH --job-name=cbsim-parity
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
ROOT=${CBSIM_ROOT:-/scratch/$USER/cbsim}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
module load apptainer 2>/dev/null || true
export HF_HOME=${HF_HOME:-/scratch/$USER/hf} HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$ROOT/repo/src VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$ROOT/results"
apptainer exec --bind "/scratch/$USER" "$SIF" python3 "$ROOT/repo/bench/scheduler_parity.py" \
  --scenarios "${SCENARIOS:-1000}" --seed "${SEED:-1000}" --out "$ROOT/results/parity_${SLURM_JOB_ID}.json"
