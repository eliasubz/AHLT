#!/usr/bin/env python3

import sys
from dataset import *
from MEM import *
from SVM import *

def predict(datafile, modelfile, outputfile):
    # load data to annotate
    ds = Dataset(datafile)

    # load trained model to use
    ext = modelfile[-4:]
    if ext == ".MEM" : model = MEM(modelfile)
    elif ext == ".SVM" : model = SVM(modelfile)
    else :
        print(f"Invalid model type '{ext}'")
        sys.exit(1)

    outf = open(outputfile, "w")
    # process each example and get predicted label
    for ex in ds.instances():
        pred = model.predict(ex["features"])
        if pred != "null" :
            print(ex["sid"],ex["e1"],ex["e2"],pred[0], sep="|", file=outf)
    outf.close()
            

    
## --------- MAIN PROGRAM ----------- 
## --
## -- Extracts Drug NE from all XML files in target-dir
## --
if __name__ == "__main__" :
    datafile = sys.argv[1]
    modelfile = sys.argv[2]
    outfile = sys.argv[3]
    
    predict(datafile, modelfile, outfile)

