#! /usr/bin/python3

import sys, os

from dataset import Dataset
from train import do_train
from predict import predict

##########################################################
#
#  This script allows to run a series of experiments
#  on DDI on medical text using NN
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

import paths
sys.path.append(paths.UTIL)
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
        os.makedirs(paths.PREPROCESS, exist_ok=True)
        print("Creating parsed test pickle file...         ")
        ds = Dataset(os.path.join(paths.DATA,"test.xml"))
        ds.save(os.path.join(paths.PREPROCESS,"test.pck"))

    else : # otherwise, extract features for train and devel
        os.makedirs(paths.PREPROCESS, exist_ok=True)
        # convert datasets to feature vectors
        print("Creating parsed train pickle file...         ")
        ds = Dataset(os.path.join(paths.DATA,"train.xml"))
        ds.save(os.path.join(paths.PREPROCESS,"train.pck"))
        print("Creating parsed devel pickle file...         ")
        ds = Dataset(os.path.join(paths.DATA,"devel.xml"))
        ds.save(os.path.join(paths.PREPROCESS,"devel.pck"))

    
# for each required model, see if training or prediction are required
   
if "train" in sys.argv[1:] :
    os.makedirs(paths.MODELS, exist_ok=True)
    # train model
    print(f"Training model {params['name']} ...")
    do_train(os.path.join(paths.PREPROCESS,"train.pck"),
             os.path.join(paths.PREPROCESS,"devel.pck"),
             params,
             os.path.join(paths.MODELS,params["name"]))
    
if "predict" in sys.argv[1:] :    
    os.makedirs(paths.RESULTS, exist_ok=True)
    if "test" in sys.argv[1:] :
        # run model on test data and evaluate results
        print(f"Running {params['name']} model on test...")
        predict(os.path.join(paths.MODELS,params["name"]),
                os.path.join(paths.PREPROCESS,"test.pck"),
                params,
                os.path.join(paths.RESULTS,"test-"+params["name"]+".out"))
        evaluate("DDI", os.path.join(paths.DATA,"test.xml"),
                 os.path.join(paths.RESULTS,"test-"+params["name"]+".out"),
                 os.path.join(paths.RESULTS,"test-"+params["name"]+".stats"))
                 
    else :
        # run model on devel data and evaluate results
        print(f"Running {params['name']} model on devel...")
        predict(os.path.join(paths.MODELS,params["name"]),
                os.path.join(paths.PREPROCESS,"devel.pck"),
                params,
                os.path.join(paths.RESULTS,"devel-"+params["name"]+".out"))
        evaluate("DDI", os.path.join(paths.DATA,"devel.xml"),
                 os.path.join(paths.RESULTS,"devel-"+params["name"]+".out"),
                 os.path.join(paths.RESULTS,"devel-"+params["name"]+".stats"))

        '''
        # run model on train data and evaluate results
        print(f"Running {params['name']} model on train...")
        predict(os.path.join(paths.MODELS,params["name"]),
                os.path.join(paths.PREPROCESS, "train.pck"),
                params,
                os.path.join(paths.RESULTS,"train-"+model+".out"))
        evaluate("DDI", os.path.join(paths.DATA,"train.xml"),
                 os.path.join(paths.RESULTS,"train-"+model+".out"),
                 os.path.join(paths.RESULTS,"train-"+model+".stats"))
        '''
