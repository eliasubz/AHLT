#! /usr/bin/python3

import sys
from os import system

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import *
from codemaps import *

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"

## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def output_entities(data, preds, codes, outfile) :

   outf = open(outfile, "w")
   for sid,tags in zip(data.sentence_ids(),preds) :
      inside = False
      text,tokens = data.get_sentence_text(sid), data.get_sentence_tokens(sid)
      for k in range(0, min(len(tokens),codes.maxlen)) :
         y = tags[k]
         tk = tokens[k]
            
         if (y[0]=="B") :
             entity_start = tk.idx
             entity_end = tk.idx + len(tk.text)
             entity_type = y[2:]
             inside = True
         elif (y[0]=="I" and inside) :
             entity_end = tk.idx + len(tk.text)
         elif (y[0]=="O" and inside) :
             print(sid, str(entity_start)+"-"+str(entity_end-1), text[entity_start:entity_end], entity_type, sep="|", file=outf)
             inside = False
        
      if inside : print(sid, str(entity_start)+"-"+str(entity_end-1), text[entity_start:entity_end], entity_type, sep="|", file=outf)

   outf.close()

#----------------------------------------------
def encode_dataset(ds, codes, params) :
   X = codes.encode_words(ds)
   if used_device == "cuda:0" :
      X = [x.to(torch.device(used_device)) for x in X]
   return DataLoader(TensorDataset(*X), params["batch_size"])


#----------------------------------------------
def predict(modelname, datafile, params, outfile) :
    # set default if not given
    if "batch_size" not in params: params["batch_size"]=16

    model = torch.load(os.path.join(modelname,"network.nn"),
                       weights_only=False,
                       map_location=torch.device(used_device))   
    model.eval()
    codes = Codemaps(os.path.join(modelname,"codemaps"), params)

    testdata = Dataset(datafile)
    test_loader = encode_dataset(testdata, codes, params)

    Y = []
    for X in test_loader:
       y = model.forward(*X)
       Y.extend([[codes.idx2label(torch.argmax(w)) for w in s] for s in y] )

    # extract & evaluate entities with basic model
    output_entities(testdata, Y, codes, outfile)

   
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py modelname datafile outfile [batch_size=N]
## --
## -- Extracts Drug NE from sentences in datafile
## --

if __name__ == "__main__" :
    modelname = sys.argv[1]
    datafile= sys.argv[2]
    outfile= sys.argv[3]

    params={}
    for p in sys.argv[4:] :
       k,v = p.split("=")
       params[k]=int(v)

    predict(modelname, datafile, params, outfile)

