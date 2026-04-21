#!/usr/bin/env bash
# Fine-tuned inference + evaluator on a split — **edit** Slurm and venv like ``FT-train.sh``.
#
# Usage:
#   sbatch FT-inference.sh llama32B3 prompts01.json devel FT-llama32B3-quant.weights -quant
#
# The ``weightdir`` argument must match the directory name produced under ``../models/`` by ``finetune-train.py``.

#SBATCH --job-name=nerc-ft-infer
#SBATCH -p cuda
#SBATCH -A YOUR_SLURM_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS="$(cd "$BIN/../../.." && pwd)"
UTIL="$LABS/util/evaluator.py"
cd "$BIN"

export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_cache}"
mkdir -p "$HF_HOME"

# shellcheck source=/dev/null
source "${LLM_VENV:-/path/to/your/course/venv}/bin/activate"

MODEL="$1"
PROMPTS="$2"
TEST="$3"
WEIGHTS="$4"
QUANT="${5:-}"

python3 finetune-inference.py "$MODEL" "$PROMPTS" "$TEST" "$WEIGHTS" "$QUANT"

SLUG="${MODEL//\//__}"
if [[ "$QUANT" == "-quant" ]]; then
  QUANT_SUFFIX="-quant"
else
  QUANT_SUFFIX=""
fi
OUT="$BIN/../results/FT-${SLUG}${QUANT_SUFFIX}-${TEST}.out"
python3 "$UTIL" NER "$LABS/data/${TEST}.xml" "$OUT" "${OUT%.out}.stats"
