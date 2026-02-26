#! /usr/bin/python3

import sys, os

from dataset import Dataset
from train import do_train
from predict import predict

##########################################################
#
#  This script allows to run a series of experiments
#  on NER on medical text using NN
#
#  You can select wich steps of the experiment execcute:
#    - parse: Use spaccy to parse the documents and store results in pickle files
#    - train: Train a NN model
#    - predict: Apply the model to development data set and evaluate performance
#
#  You can add hyperparameters for training
#    - batch_size, max_len, suf_len
#    Omitted parameters will receive a default value
#    Parametres may be mixed, each model will select its own.
#
#  Examples:
#
#      # Extract features, train, and evaluate a CRF model
#      python3 run.py parse
#      python3 run.py train name=mymodel_001 batch_size=32 max_len=140 epochs=5
#      python3 run.py predict name=mymodel_001
#
#      # the 3 lines above can be run in a single one:
#      python3 run.py parse train predict name=mymodel_001 batch_size=32 max_len=140
##

BINDIR=os.path.abspath(os.path.dirname(__file__)) # location of this file
NERDIR=os.path.dirname(BINDIR) # one level up
SOLDIR=os.path.dirname(NERDIR) # one level up
MAINDIR=os.path.dirname(SOLDIR) # one level up
DATADIR=os.path.join(MAINDIR,"data") # down to "data"
UTILDIR=os.path.join(MAINDIR,"util") # down to "util"

sys.path.append(UTILDIR)
from evaluator import evaluate

# extract training hyperparameters from command line
print("read params")
params = {}
for p in sys.argv[1:]:
    if "=" in p:
        par,val = p.split("=")
        params[par] = val
        
if "name" not in params: params["name"]="mymodel_000"

# if feature extraction is required, do it
if "parse" in sys.argv[1:] :
    # if test is required, extract features from test
    if "test" in sys.argv[1:] : 
        os.makedirs(os.path.join(NERDIR, "preprocessed"), exist_ok=True)
        print("Creating parsed test pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"test.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","test.pck"))

    else : # otherwise, extract features for train and devel
        os.makedirs(os.path.join(NERDIR, "preprocessed"), exist_ok=True)
        # convert datasets to feature vectors
        print("Creating parsed train pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"train.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","train.pck"))
        print("Creating parsed devel pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"devel.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","devel.pck"))

    
# for each required model, see if training or prediction are required
   
if "train" in sys.argv[1:] :
    os.makedirs(os.path.join(NERDIR,"models"), exist_ok=True)
    # train model
    print(f"Training model {params['name']} ...")
    do_train(os.path.join(NERDIR, "preprocessed","train.pck"),
             os.path.join(NERDIR, "preprocessed","devel.pck"),
             params,
             os.path.join(NERDIR,"models",params["name"]))
    
if "predict" in sys.argv[1:] :    
    os.makedirs(os.path.join(NERDIR,"results"), exist_ok=True)
    if "test" in sys.argv[1:] :
        # run model on test data and evaluate results
        print(f"Running {params['name']} model on test...")
        predict(os.path.join(NERDIR,"models",params["name"]),
                os.path.join(NERDIR, "preprocessed","test.pck"),
                params,
                os.path.join(NERDIR,"results","test-"+params["name"]+".out"))
        evaluate("NER", os.path.join(DATADIR,"test.xml"),
                 os.path.join(NERDIR,"results","test-"+params["name"]+".out"),
                 os.path.join(NERDIR,"results","test-"+params["name"]+".stats"))
                 
    else :
        # run model on devel data and evaluate results
        print(f"Running {params['name']} model on devel...")
        predict(os.path.join(NERDIR,"models",params["name"]),
                os.path.join(NERDIR, "preprocessed","devel.pck"),
                params,
                os.path.join(NERDIR,"results","devel-"+params["name"]+".out"))
        evaluate("NER", os.path.join(DATADIR,"devel.xml"),
                 os.path.join(NERDIR,"results","devel-"+params["name"]+".out"),
                 os.path.join(NERDIR,"results","devel-"+params["name"]+".stats"))

        '''
        # run model on train data and evaluate results
        print(f"Running {params['name']} model on train...")
        predict(os.path.join(NERDIR, "models",params["name"]),
                os.path.join(NERDIR, "preprocessed", "train.pck"),
                params,
                os.path.join(NERDIR,"results","train-"+model+".out"))
        evaluate("NER", os.path.join(DATADIR,"train.xml"),
                 os.path.join(NERDIR,"results","train-"+model+".out"),
                 os.path.join(NERDIR,"results","train-"+model+".stats"))
        '''
