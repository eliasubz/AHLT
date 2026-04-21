#!/usr/bin/env bash
# =============================================================================
# Phase 1 (few-shot): one Slurm job = full **devel** loop the PDF describes for FS:
#   train.xml → few-shot pool → annotate **devel** → evaluator → TSV log row.
#
# Optional: also score **train** (same k) for qualitative error analysis only —
#   do **not** use train scores to pick hyperparameters; **devel** is for selection.
#
# Submit (after editing #SBATCH and LLM_VENV):
#   cd .../1.3.NERC-LLM/bin
#   export MODEL=llama32B3
#   export LLM_MODEL_ROOT=/scratch/$USER/hf_models
#   sbatch phase1_fewshot_devel_loop.sh
#
# Override with env (examples):
#   SHOTS_LIST="3 5"  PROMPTS_JSON=$PWD/prompts01.json  QUANT_FLAG=-quant
#   PHASE1_EVAL_TRAIN=1   # adds a train split pass per k (heavier)
# =============================================================================

#SBATCH --job-name=nerc-p1-fs-devel
#SBATCH -p cuda
#SBATCH -A YOUR_SLURM_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
##SBATCH --output=%x-%j.out
##SBATCH --error=%x-%j.err

set -euo pipefail

BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS="$(cd "$BIN/../../.." && pwd)"
UTIL="$LABS/util/evaluator.py"
DEVEL_XML="$LABS/data/devel.xml"
TRAIN_XML="$LABS/data/train.xml"
cd "$BIN"

export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_cache}"
mkdir -p "$HF_HOME"

# shellcheck source=/dev/null
source "${LLM_VENV:-/path/to/your/course/venv}/bin/activate"

MODEL="${MODEL:-llama32B3}"
PROMPTS_JSON="${PROMPTS_JSON:-$BIN/prompts01.json}"
SHOTS_LIST="${SHOTS_LIST:-3 5 10}"
TRAIN_STEM="${TRAIN_STEM:-train}"
QUANT_FLAG="${QUANT_FLAG:--quant}"
PHASE1_EVAL_TRAIN="${PHASE1_EVAL_TRAIN:-0}"

SLUG="${MODEL//\//__}"
if [[ "$QUANT_FLAG" == "-quant" ]]; then
  QUANT_SUFFIX="-quant"
else
  QUANT_SUFFIX=""
fi

run_one_split() {
  local k="$1"
  local split="$2"
  local gold_xml="$3"

  echo "========== fewshot  k=${k}  test=${split}  model=${MODEL} ==========" >&2
  python3 fewshot.py "$MODEL" "$PROMPTS_JSON" "$k" "$TRAIN_STEM" "$split" "$QUANT_FLAG"

  local out="$BIN/../results/FS-${SLUG}-${k}-${split}${QUANT_SUFFIX}.out"
  local stats="${out%.out}.stats"

  python3 "$UTIL" NER "$gold_xml" "$out" "$stats"

  local score=""
  if [[ -f "$stats" ]]; then
    # Micro-averaged row is ``m.avg`` (padded); skip ``m.avg(no class)``.
    score="$(awk -F'\t' '$1 ~ /^m.avg/ && $1 !~ /no class/ {print $NF; exit}' "$stats" | tr -d '\r' || true)"
  fi

  python3 experiment_log.py \
    --strategy fewshot \
    --model "$MODEL" \
    --prompts "$PROMPTS_JSON" \
    --split "$split" \
    --k "$k" \
    --quant "${QUANT_FLAG#-}" \
    --score "$score" \
    --outfile "$out" \
    --notes "phase1_fewshot_devel_loop.sh stats=${stats}"
}

for K in $SHOTS_LIST; do
  run_one_split "$K" "devel" "$DEVEL_XML"
done

if [[ "$PHASE1_EVAL_TRAIN" == "1" ]]; then
  echo "========== PHASE1_EVAL_TRAIN=1  (train scores for analysis only) ==========" >&2
  for K in $SHOTS_LIST; do
    run_one_split "$K" "train" "$TRAIN_XML"
  done
fi

echo "Phase 1 few-shot loop finished. Logs: $BIN/../results/EXPERIMENT_LOG.tsv" >&2
