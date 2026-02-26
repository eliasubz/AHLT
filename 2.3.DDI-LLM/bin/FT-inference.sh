#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 


## Usage: 
##    sbatch FT-inference.sh llama32B3 prompt01 devel FT-llama32B3.weights [-quant]

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

MODEL=$1
PROMPTS=$2
TEST=$3
WEIGHTS=$4
QUANT=$5

python3 finetune-inference.py $MODEL $PROMPTS $TEST $WEIGHTS $QUANT
if (test $? != 0); then exit; fi

python3 ../../../util/evaluator.py DDI ../../../data/$TEST.xml  ../results/FT-${MODEL}${QUANT}-${TEST}.out ../results/FT-${MODEL}${QUANT}-${TEST}.stats

deactivate
