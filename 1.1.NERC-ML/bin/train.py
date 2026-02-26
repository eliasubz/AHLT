#!/usr/bin/env python3

import sys
from MEM import *
from SVM import *
from CRF import *

def train(datafile, params, modelfile) :
    # Create an empty model of the appropriate type
    ext = modelfile[-4:].lower()
    if ext == ".mem" : model = MEM(modelfile, params)
    elif ext == ".svm" : model = SVM(modelfile, params)
    elif ext == ".crf" : model = CRF(modelfile, params)        
    else :
        print(f"Invalid model type '{ext}'")
        sys.exit(1)

    # Train and store the model
    model.train(datafile)


if __name__ == "__main__" :
    # get file where model will be written
    datafile = sys.argv[1]
    modelfile = sys.argv[2]
    
    # get parameters in line.  e.g. C=10 kernel=rbf degree=2
    params = {}
    pars = sys.argv[3:]
    for x in pars:
        par,val = x.split("=")
        params[par] = val

    # train model and store in given filename
    train(datafile, params, modelfile)
