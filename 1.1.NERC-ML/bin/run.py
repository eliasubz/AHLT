#! /usr/bin/python3

import sys, os

from extract_features import extract_features, parse_feature_flags_param
from train import train
from predict import predict
from dictionaries import Dictionaries
import paths

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
#  Optional run isolation and feature ablation:
#    - run_id=<slug>  -> preprocessed/runs/<slug>/, models/runs/<slug>/, results/runs/<slug>/
#    - feature_flags=a,b,c  -> only listed ablation groups ON (others OFF).
#        Omit feature_flags entirely -> all groups ON (default).
#        feature_flags=  (empty) -> all groups OFF (base profile).
#      Keys: med_patterns, stopwords, external_phrase, affix5_alpha, length_extras, length_next
#
#  You can add hyperparameters for each of the algorithms training
#    - common: top_features_file, top_features_k
#              (defaults: results/devel.template_mi.tsv and 100)
#    - for CRF: algorithm, feature.minfreq, c1, c2, max_iterations, epsilon
#               drug_n_oversample (repeat factor for sequences with B/I-drug_n)
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
#      # Extract features, train, and evaluate a CRF model (context_window is required)
#      python3 run.py extract train predict CRF context_window=1
#      python3 run.py train predict CRF context_window=1
#
#      # Isolated run with metrics paths under results/runs/<id>/
#      python3 run.py extract train predict CRF context_window=1 run_id=exp01
#
#      # Ablation: only med_patterns + stopwords on, rest off
#      python3 run.py extract train predict CRF context_window=1 run_id=exp02 feature_flags=med_patterns,stopwords
#
#      # Full automated grid (base, singletons, cumulative ladder); from bin/:
#      python3 sweep_runs.py
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

from evaluator import evaluate

# extract training hyperparameters from command line
print("read params")
params = {}
for p in sys.argv[1:]:
    if "=" in p:
        par,val = p.split("=")
        params[par] = val

run_id = (params.get("run_id") or "").strip() or None
feature_flags = parse_feature_flags_param(params.get("feature_flags"))

if run_id:
    run_preprocess_dir = os.path.join(paths.PREPROCESS, "runs", run_id)
    run_models_dir = os.path.join(paths.MODELS, "runs", run_id)
    run_results_dir = os.path.join(paths.RESULTS, "runs", run_id)
else:
    run_preprocess_dir = paths.PREPROCESS
    run_models_dir = paths.MODELS
    run_results_dir = paths.RESULTS

train_feat = os.path.join(run_preprocess_dir, "train.feat")
devel_feat = os.path.join(run_preprocess_dir, "devel.feat")
test_feat = os.path.join(run_preprocess_dir, "test.feat")

# if creting dictionaries is required, do it
if "dicts" in sys.argv[1:] :
   print("Creating dictionaries")
   dict = Dictionaries()
   dict.save(os.path.join(paths.RESOURCES,"dictionaries"))

# if feature extraction is required, do it
if "extract" in sys.argv[1:] :
    if "context_window" not in params:
        raise ValueError("context_window must be specified, e.g. context_window=1")
    context_window = int(params["context_window"])
    print(f"Using context_window={context_window} for feature extraction")
    if run_id:
        print(f"run_id={run_id} (isolated preprocessed/models/results)")
    os.makedirs(run_preprocess_dir, exist_ok=True)

    # if test is required, extract features from test
    if "test" in sys.argv[1:] :
        print("Extracting features for test...")
        extract_features(
            os.path.join(paths.DATA, "test.xml"),
            test_feat,
            context_window,
            feature_flags,
        )

    else : # otherwise, extract features for train and devel
        print("Extracting features for train...")
        extract_features(
            os.path.join(paths.DATA, "train.xml"),
            train_feat,
            context_window,
            feature_flags,
        )
        print("Extracting features for devel...")
        extract_features(
            os.path.join(paths.DATA, "devel.xml"),
            devel_feat,
            context_window,
            feature_flags,
        )

    
# for each required model, see if training or prediction are required
for model in ["CRF", "SVM", "MEM"] :
    if model not in sys.argv[1:] : continue
   
    if "train" in sys.argv[1:] :
        os.makedirs(run_models_dir, exist_ok=True)
        # train model
        print(f"Training {model} model...")
        modelfile = os.path.join(run_models_dir, "model." + model)
        train(train_feat, params, modelfile)
        
    if "predict" in sys.argv[1:] :    
        os.makedirs(run_results_dir, exist_ok=True)
        modelfile = os.path.join(run_models_dir, "model." + model)
        if "test" in sys.argv[1:] :
            if "test" in sys.argv[1:] :
                # run model on test data and evaluate results
                print(f"Running {model} model...")
                test_out = os.path.join(run_results_dir, "test-" + model + ".out")
                test_stats = os.path.join(run_results_dir, "test-" + model + ".stats")
                predict(test_feat, modelfile, test_out)
                evaluate("NER", 
                         os.path.join(paths.DATA,"test.xml"),
                         test_out,
                         test_stats)
                         
        else :
            # run model on devel data and evaluate results
            print(f"Running {model} model...")
            devel_out = os.path.join(run_results_dir, "devel-" + model + ".out")
            devel_stats = os.path.join(run_results_dir, "devel-" + model + ".stats")
            predict(devel_feat, modelfile, devel_out)
            evaluate("NER", 
                     os.path.join(paths.DATA,"devel.xml"),
                     devel_out,
                     devel_stats)

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
