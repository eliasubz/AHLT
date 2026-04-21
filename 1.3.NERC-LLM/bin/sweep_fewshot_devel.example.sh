#!/usr/bin/env bash
# Example few-shot sweep on **devel** (use **train** only for qualitative error analysis).
# Copy to ``sweep_fewshot_devel.sh``, set MODEL / venv / ``LLM_MODEL_ROOT`` / ``HF_HOME``, then run on GPU.
set -euo pipefail

BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS="$(cd "$BIN/../../.." && pwd)"
UTIL="$LABS/util/evaluator.py"
DEVEL_XML="$LABS/data/devel.xml"
cd "$BIN"

MODEL="${MODEL:-llama32B3}"
PROMPTS_JSON="${PROMPTS_JSON:-$BIN/prompts01.json}"
SHOTS_LIST="${SHOTS_LIST:-3 5 10}"
QUANT_FLAG="${QUANT_FLAG:--quant}"

# Match ``paths.model_slug`` for output filenames: hub slashes -> "__"
SLUG="${MODEL//\//__}"
if [[ "$QUANT_FLAG" == "-quant" ]]; then
  QUANT_SUFFIX="-quant"
else
  QUANT_SUFFIX=""
fi

for K in $SHOTS_LIST; do
  python3 fewshot.py "$MODEL" "$PROMPTS_JSON" "$K" train devel "$QUANT_FLAG"
  OUT="$BIN/../results/FS-${SLUG}-${K}-devel${QUANT_SUFFIX}.out"
  python3 "$UTIL" NER "$DEVEL_XML" "$OUT" "${OUT%.out}.stats" || true
  python3 experiment_log.py \
    --strategy fewshot \
    --model "$MODEL" \
    --prompts "$PROMPTS_JSON" \
    --split devel \
    --k "$K" \
    --quant "${QUANT_FLAG#-}" \
    --outfile "$OUT" \
    --notes "sweep_fewshot_devel.example.sh"
done
