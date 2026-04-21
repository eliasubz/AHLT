import os,sys,time,json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paths
from model import Inference
from prompts import Prompts
from examples import Examples

# ------------ check command line and get arguments -----------------
def get_arguments():
    valid_flags = {"-quant", "-ollama", "-diverse"}
    flags    = {a for a in sys.argv[1:] if a.startswith("-")}
    pos_args = [a for a in sys.argv[1:] if not a.startswith("-")]

    unknown = flags - valid_flags
    if unknown or len(pos_args) != 5:
        print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile"
              f" [-quant] [-ollama] [-diverse]", file=sys.stderr)
        sys.exit(1)

    model        = pos_args[0]
    promptfile   = pos_args[1]
    num_few_shot = int(pos_args[2])
    traindata    = pos_args[3]
    testdata     = pos_args[4]
    quantized    = "-quant"   in flags
    ollama       = "-ollama"  in flags
    diverse      = "-diverse" in flags

    return model, promptfile, num_few_shot, traindata, testdata, quantized, ollama, diverse


############## main ###################

# get command line arguments
model, promptfile, num_few_shot, traindata, testdata, quantized, ollama, diverse = get_arguments()

print(f"========= FEW SHOT === PROMPTS={promptfile}  SHOTS={num_few_shot}  DATA={testdata}"
      f"  quantized={quantized}  diverse={diverse}", file=sys.stderr)

# load training data (FS examples)
trainfile = os.path.join(paths.DATA,traindata+".xml")
strategy  = "diverse" if diverse else "random"
fs_examples = Examples(trainfile, "NER").select_examples(num_few_shot, strategy=strategy)

# load prompts, create few-shot prompt
prompts = Prompts(promptfile, fs_examples)

# load test data
testfile = os.path.join(paths.DATA,testdata+".xml")
test = Examples(testfile, "NER")

# load model and tokenizer
t0 = time.time()
if ollama:
   engine = Inference(model, ollama=True)
else :
   MODEL_PATH = paths.resolve_llm_model_path(model)
   engine = Inference(MODEL_PATH, quantized=quantized)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# annotate each example in testdata
t0 = time.time()
annotated = []
for i,ex in enumerate(test.select_examples()):
    print(f"Processing example {i} - {ex['id']}", flush=True, file=sys.stderr)
    
    # create prompt for this example, adding it to FS prompt
    messages = prompts.prepare_messages(ex['input'])
    # call model to generate response 
    gen_text = engine.generate(messages)
    # store responses
    ex['predicted'] = gen_text
    ex['evaluator'] = test.eval_format(ex,gen_text)
    annotated.append(ex)

print("Done", file=sys.stderr)
print(f"Processed {len(annotated)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(annotated):.2f} sec/example)", file=sys.stderr)

os.makedirs(paths.RESULTS, exist_ok=True)
quant = "-quant" if quantized else ""
slug = paths.model_slug(model)
outfname = os.path.join(paths.RESULTS,
                        f"FS-{slug}-{num_few_shot}-{testdata}{quant}")
with open(outfname+".json", "w") as of:  
   json.dump(annotated, of, indent=1, ensure_ascii=False)
with open(outfname+".out", "w") as of:  
   for e in annotated:
      if e["evaluator"]: 
          print("\n".join(e["evaluator"]), file=of)

# clean up gpu
del engine
torch.cuda.empty_cache() 

