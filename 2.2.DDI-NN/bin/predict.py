#! /usr/bin/python3

import sys
from os import system

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import *
from codemaps import *

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"

## --------- DDI extractor ----------- 
def output_interactions(data, preds, outfile) :
   outf = open(outfile, "w")
   for exmp,tag in zip(data.sentences(),preds) :
      if tag!='null' :
         print(exmp['sid'], exmp['e1'], exmp['e2'], tag, sep="|", file=outf)
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
    # run each validation example and report validation loss
    for X in test_loader:
       # X is a list of input tensors (no labels were loaded in the dataloader)
       y = model.forward(*X) # run example through the network
       # add results to result list
       Y.extend([codes.idx2label(torch.argmax(s)) for s in y])

    # extract relations from result list
    output_interactions(testdata, Y, outfile)


   
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

