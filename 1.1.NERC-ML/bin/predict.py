#!/usr/bin/env python3

import sys
from MEM import *
from SVM import *
from CRF import *

# --------------------------------------------------
# extract identified drugs according to BIO tags for each word.

def output_entities(toks, predictions, outf) :
    inside = False;
    for k in range(len(predictions)) :
        y = predictions[k]
        (sid, form, offS, offE) = toks[k]

        if (y[0]=="B") :
            entity_form = form
            entity_start = offS
            entity_end = offE
            entity_type = y[2:]
            inside = True
        elif (y[0]=="I" and inside) :
            entity_form += " "+form
            entity_end = offE
        elif (y[0]=="O" and inside) :
            print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|", file=outf)
            inside = False

    if inside : print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|", file=outf)

    
def predict(datafile, modelfile, outputfile):
    
    # load data to annotate
    ds = Dataset(datafile)

    # load trained model to use
    ext = modelfile[-4:].lower()
    if ext == ".mem" : model = MEM(modelfile)
    elif ext == ".svm" : model = SVM(modelfile)
    elif ext == ".crf" : model = CRF(modelfile)
    else :
        print(f"Invalid model type '{ext}'")
        sys.exit(1)
        
    # open outfile
    outf = open(outputfile, "w")
    
    for xseq,_,toks in ds.instances():
        # process each sentence
        # each word has a list of features (xseq) for the prediction
        # plus positional info (toks) to format the output

        # get BIO labels for each word in the sentence
        predictions = model.predict(xseq)
        # Convert BIO labels to drugs
        output_entities(toks, predictions, outf)

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

