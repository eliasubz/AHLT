import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paths
from model import Inference
from examples import Examples
from prompts import Prompts

# ------------ check command line and get arguments -----------------
def get_arguments():
    if not 5<=len(sys.argv)<=6  or (len(sys.argv)==6 and sys.argv[5]!="-quant"):
        print(f"Usage:  {sys.argv[0]} model prompts testfile weightdir [-quant]", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    promptfile = sys.argv[2]
    testdata = sys.argv[3]
    weightdir = sys.argv[4]
    quantized = (len(sys.argv)==6)

    if quantized and "-quant" not in weightdir:
        print("WARNING: Loading adapters for non-quantized models into a quantized model will result in erratic model output.")
    if not quantized and "-quant" in weightdir:
        print("WARNING: Loading adapters for quantized models into a non-quantized model will result in erratic model output.")

    weightdir = os.path.join(paths.MODELS, weightdir)

    return model, promptfile, testdata, weightdir, quantized

    
############ MAIN ##################

# get command line arguments
model, promptfile, testdata, weightdir, quantized =  get_arguments()
print(f"========= FT inference === MODEL={model}  WEIGHTS={weightdir}  quantized={quantized}")

# load prompts
prompts = Prompts(promptfile)

# load test/devel dataset
testfile = os.path.join(paths.DATA,testdata+".xml")
test = Examples(testfile, "NER")

# load model and tokenizer
t0 = time.time()
MODEL_PATH = f"/scratch/nas/1/PDI/mml0/models/{model}"
engine = Inference(MODEL_PATH, quantized=quantized, peft=weightdir)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# analyze each example
t0 = time.time()
annotated = []
for i,ex in enumerate(test.select_examples()):
    print(f"*** Processing example {i}", flush=True)
    # prepare sequence of messages for this example
    messages = prompts.prepare_messages(ex['input'])    
    # call model to generate response            
    gen_text = engine.generate(messages)
    # extract json from response
    ex["predicted"] = gen_text
    ex['evaluator'] = test.eval_format(ex,gen_text)
    annotated.append(ex)


print("Done")
print(f"Processed {len(annotated)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(annotated):.2f} sec/example)")

# save output
os.makedirs(paths.RESULTS, exist_ok=True)
quant = "-quant" if quantized else ""
outfname = os.path.join(paths.RESULTS,
                        f"FT-{model}{quant}-{testdata}")
with open(outfname+".json", "w") as of:
   json.dump(annotated, of, indent=1, ensure_ascii=False)
with open(outfname+".out", "w") as of:  
   for e in annotated:
      if e["evaluator"]: 
          print("\n".join(e["evaluator"]), file=of)

# clean up gpu
del engine
torch.cuda.empty_cache() 


