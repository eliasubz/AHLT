#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 4
#SBATCH --mem=64Gb 


## Usage: 
##    sbatch FT-train.sh llama32B3 prompt01 train devel [-quant]

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

MODEL=$1
PROMPTS=$2
TRAIN=$3
TEST=$4
QUANT=$5

python3 finetune-train.py $MODEL $PROMPTS $TRAIN $TEST $QUANT
if (test $? != 0); then exit; fi

deactivate
