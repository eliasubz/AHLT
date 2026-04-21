#!/usr/bin/env bash
# Few-shot NERC on GPU via Slurm — **edit** Slurm and venv for your university cluster.
#
# Usage:
#   sbatch fewshot.sh llama32B3 prompts01.json 15 train devel -quant
#
# ``prompts`` should be a path to a JSON file (e.g. ``$PWD/prompts01.json`` when submitting from ``bin/``).

#SBATCH --job-name=nerc-fewshot
#SBATCH -p cuda
#SBATCH -A YOUR_SLURM_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH --time=08:00:00

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
SHOTS="$3"
TRAIN="$4"
TEST="$5"
QUANT="$6"

python3 fewshot.py "$MODEL" "$PROMPTS" "$SHOTS" "$TRAIN" "$TEST" "$QUANT"

SLUG="${MODEL//\//__}"
if [[ "$QUANT" == "-quant" ]]; then
  QUANT_SUFFIX="-quant"
else
  QUANT_SUFFIX=""
fi
OUT="$BIN/../results/FS-${SLUG}-${SHOTS}-${TEST}${QUANT_SUFFIX}.out"
python3 "$UTIL" NER "$LABS/data/${TEST}.xml" "$OUT" "${OUT%.out}.stats"
