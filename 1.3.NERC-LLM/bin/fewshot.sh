#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb


## Usage:
##   sbatch fewshot.sh MODEL PROMPTS SHOTS TRAIN TEST [-quant] [-ollama] [-diverse]
##
## Examples:
##   sbatch fewshot.sh llama3.2:3b prompts01 10 train devel -ollama
##   sbatch fewshot.sh llama3.2:3b prompts01 10 train devel -ollama -diverse
##   sbatch fewshot.sh llama3.2-3b prompts01 10 train devel -quant

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

MODEL=$1
PROMPTS=$2
SHOTS=$3
TRAIN=$4
TEST=$5
shift 5
FLAGS="$@"   # remaining args: any combination of -quant, -ollama, -diverse

# derive filename components (must match fewshot.py logic)
MODELBASE=$(echo $MODEL | tr ':/' '__')
PROMPTBASE=$(basename $PROMPTS .json)
QUANT="";  DIVERSE=""
for f in $FLAGS; do
    [ "$f" = "-quant" ]   && QUANT="-quant"
    [ "$f" = "-diverse" ] && DIVERSE="-diverse"
done

python3 fewshot.py $MODEL $PROMPTS $SHOTS $TRAIN $TEST $FLAGS
if [ $? -ne 0 ]; then deactivate; exit 1; fi

OUTBASE="../results/FS-${MODELBASE}-${PROMPTBASE}-${SHOTS}-${TEST}${DIVERSE}${QUANT}"
python3 ../../../util/evaluator.py NER ../../../data/${TEST}.xml \
    ${OUTBASE}.out ${OUTBASE}.stats

deactivate
