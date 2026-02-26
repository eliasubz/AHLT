#! /usr/bin/python3

import sys, os

from extract_features import extract_features
from train import train
from predict import predict
from dictionaries import Dictionaries

##########################################################
#
#  This script allows to run a series of experiments
#  on NER on medical text
#
#  You can train and test different ML algorithms: CRF, MEM, SVM
#
#  You can select which steps of the experiment to execute:
#    - dicts: Create dictionaries useful to extract features using data in resources dir.
#    - extract: extract features to convert text tokens to feature vectors
#    - train: Train a ML model
#    - predict: Apply the model to development data set and evaluate performance
#
#  You can add hyperparameters for each of the algorithms training
#    - for CRF: algorithm, feature.minfreq, c1, c2, max_iterations, epsilon
#               More details about parameters at:
#               https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
#    - for MEM: C, solver, max_iter, n_jobs
#               More details about parameters at:
#               https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#               sklearn.linear_model.LogisticRegression page
#    - for SVM: C, kernel, degree, gamma
#               More details about parameters at: 
#               https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#    Omitted parameters will receive a default value
#    Parametres may be mixed, each model will select its own.
#
#  Examples:
#
#      # Extract features, train, and evaluate a CRF model
#      python3 run.py extract
#      python3 run.py train CRF max_iterations=50
#      python3 run.py predict CRF
#
#      # the 3 lines above can be run in a single one:
#      python3 run.py extract train predict CRF max_iterations=50
#
#      # Extract train, and evaluate a SVM model (assumig features were already extracted)
#      python3 run.py train predict SVM C=10 kernel=rbf
#
#      # several models can be trained/evaluated in one command
#      # The line below will do the same than all the preceeding lines
#      python3 run.py extract train predict CRF SVM C=10 kernel=rbf max_iterations=50
#
#      # the order of the arguments is not relevant, so the line below is equivalent to the previous one
#      python3 run.py kernel=rbf CRF extract C=10 predict SVM max_iterations=50 train
#

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

# if creting dictionaries is required, do it
if "dicts" in sys.argv[1:] :
   print("Creating dictionaries")
   dict = Dictionaries()
   dict.save(os.path.join(paths.RESOURCES,"dictionaries"))

# if feature extraction is required, do it
if "extract" in sys.argv[1:] :
    # if test is required, extract features from test
    if "test" in sys.argv[1:] :
        print("Extracting features for test...")
        extract_features(os.path.join(paths.DATA,"test.xml"), 
                         os.path.join(paths.PREPROCESS,"test.feat"))

    else : # otherwise, extract features for train and devel
        os.makedirs(paths.PREPROCESS, exist_ok=True)
        # convert datasets to feature vectors
        print("Extracting features for train...")
        extract_features(os.path.join(paths.DATA,"train.xml"),
                         os.path.join(paths.PREPROCESS,"train.feat"))
        print("Extracting features for devel...")
        extract_features(os.path.join(paths.DATA,"devel.xml"), 
                         os.path.join(paths.PREPROCESS,"devel.feat"))

    
# for each required model, see if training or prediction are required
for model in ["CRF", "SVM", "MEM"] :
    if model not in sys.argv[1:] : continue
   
    if "train" in sys.argv[1:] :
        os.makedirs(paths.MODELS, exist_ok=True)
        # train model
        print(f"Training {model} model...")
        train(os.path.join(paths.PREPROCESS,"train.feat"), params,
              os.path.join(paths.MODELS,"model."+model))
        
    if "predict" in sys.argv[1:] :    
        os.makedirs(paths.RESULTS, exist_ok=True)
        if "test" in sys.argv[1:] :
            if "test" in sys.argv[1:] :
                # run model on test data and evaluate results
                print(f"Running {model} model...")
                predict(os.path.join(paths.PREPROCESS,"test.feat"),
                        os.path.join(paths.MODELS,"model."+model),
                        os.path.join(paths.RESULTS,"test-"+model+".out"))
                evaluate("NER", 
                         os.path.join(paths.DATA,"test.xml"),
                         os.path.join(paths.RESULTS,"test-"+model+".out"),
                         os.path.join(paths.RESULTS,"test-"+model+".stats"))
                         
        else :
            # run model on devel data and evaluate results
            print(f"Running {model} model...")
            predict(os.path.join(paths.PREPROCESS,"devel.feat"),
                   os.path.join(paths.MODELS,"model."+model),
                   os.path.join(paths.RESULTS,"devel-"+model+".out"))
            evaluate("NER", 
                     os.path.join(paths.DATA,"devel.xml"),
                     os.path.join(paths.RESULTS,"devel-"+model+".out"),
                     os.path.join(paths.RESULTS,"devel-"+model+".stats"))

            '''
            # run model on train data and evaluate results
            print(f"Running {model} model...")
            predict(os.path.join(paths.PREPROCESS,"train.feat"),
                    os.path.join(paths.MODELS,"model."+model),
                    os.path.join(paths.RESULTS,"train-"+model+".out"))
            evaluate("NER", 
                     os.path.join(paths.DATA,"train.xml"),
                     os.path.join(paths.RESULTS,"train-"+model+".out"),
                     os.path.join(paths.RESULTS,"train-"+model+".stats"))
            '''
