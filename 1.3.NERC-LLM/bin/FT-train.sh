#!/usr/bin/env bash
# Fine-tune (LoRA) on GPU via Slurm — **edit** partition/account/QoS/GPU, venv, and wall time for your site.
#
# Usage (from this directory, after ``chmod +x FT-train.sh``):
#   sbatch FT-train.sh llama32B3 prompts01.json train devel -quant
#
# Environment (optional):
#   LLM_VENV         Path to ``bin/activate`` (default placeholder below)
#   HF_HOME          HuggingFace cache (defaults to ``$SLURM_TMPDIR/hf_cache`` on Slurm)
#   LLM_MODEL_ROOT   Directory of locally downloaded snapshots (see ``paths.resolve_llm_model_path``)

#SBATCH --job-name=nerc-ft-train
#SBATCH -p cuda
#SBATCH -A YOUR_SLURM_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BIN"

export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_cache}"
mkdir -p "$HF_HOME"

# shellcheck source=/dev/null
source "${LLM_VENV:-/path/to/your/course/venv}/bin/activate"

MODEL="$1"
PROMPTS="$2"
TRAIN="$3"
VAL="$4"
QUANT="${5:-}"

python3 finetune-train.py "$MODEL" "$PROMPTS" "$TRAIN" "$VAL" "$QUANT"
