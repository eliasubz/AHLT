import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import paths
from model import FineTuning
from examples import Examples
from prompts import Prompts

# ------------ check command line and get arguments -----------------
def get_arguments():
    if not 5<=len(sys.argv)<=6  or (len(sys.argv)==6 and sys.argv[5]!="-quant"):
        print(f"Usage:  {sys.argv[0]} model prompts trainfile valfile [-quant]", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    promptfile = sys.argv[2]
    traindata = sys.argv[3]
    valdata = sys.argv[4]
    quantized = (len(sys.argv)==6)

    return model, promptfile, traindata, valdata, quantized



############## MAIN ################

# get command line arguments
model, promptfile, traindata, valdata, quantized = get_arguments()
print(f"========= FINE TUNE == MODEL={model}  quantized={quantized}", file=sys.stderr)

# load prompts
prompts = Prompts(promptfile)

# load model and tokenizer
t0 = time.time()
MODEL_PATH = f"/scratch/nas/1/PDI/mml0/models/{model}"
engine = FineTuning(MODEL_PATH, quantized=quantized)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# load and tokenize datasets
t0 = time.time()
trainfile = os.path.join(paths.DATA,traindata+".xml")
train_examples = Examples(trainfile, "NER").select_examples()
train_dataset = engine.tokenize_dataset(train_examples, prompts)

valfile = os.path.join(paths.DATA,valdata+".xml")
val_examples = Examples(valfile, "NER").select_examples()
val_dataset = engine.tokenize_dataset(val_examples, prompts)
print(f"Dataset loading took {time.time()-t0:.1f} seconds", file=sys.stderr)
        
# Fine-tune the model and save results
t0 = time.time()
os.makedirs(paths.MODELS, exist_ok=True)
quant="-quant" if quantized else ""
outputdir = os.path.join(paths.MODELS, f"FT-{model}{quant}.weights")
engine.train(train_dataset,
             val_dataset, 
             outputdir) 
print(f"Training took {time.time()-t0:.1f} seconds", file=sys.stderr)

print("Fine-tuning complete!", file=sys.stderr)

# clean up gpu
del engine
torch.cuda.empty_cache() 

