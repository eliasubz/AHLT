#####################################################
## Class to store an ngram ME model
#####################################################

import sys
import os
import pycrfsuite
from dataset import *


class CRF:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile=None, params=None):

        self.modelfile = modelfile
        if params is None:
            # only modelfile given, assume it is an existing model and load it        
            # modelfile given, assume it is an existing model and load it
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.modelfile)
                
        else :  # params given, create new empty model

            # extract parameters if provided. Use default if not
            alg = params['algorithm'] if 'algorithm' in params else 'lbfgs'
            minf = int(params['feature.minfreq']) if 'feature.minfreq' in params else 1
            maxit =  int(params['max_iterations']) if 'max_iterations' in params else 300
            c1 = float(params['c1']) if 'c1' in params else 0.1
            c2 = float(params['c2']) if 'c2' in params else 1.0
            eps = float(params['epsilon']) if 'epsilon' in params else 0.00001
            self.top_features_file = params['top_features_file'] if 'top_features_file' in params else None
            self.top_features_k = int(params['top_features_k']) if 'top_features_k' in params else 100
            # Repeat sequences containing drug_n labels to mimic class weighting.
            self.drug_n_oversample = int(params['drug_n_oversample']) if 'drug_n_oversample' in params else 1
            if self.drug_n_oversample < 1:
                self.drug_n_oversample = 1
            # select needed parametes depending on the agorithm
            params = {'feature.minfreq' : minf, 'max_iterations' : maxit}
            if alg == "lbfgs" : params['c1'] = c1
            if alg in ["lbfgs", "l2sgd"] : params['c2'] = c2
            if alg != "l2sgd" : params['epsilon'] = eps
            # create and train empty classifier with given algorithm and parameters
            self.trainer = pycrfsuite.Trainer(alg, params)

    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile):
        allowed_features = None
        if hasattr(self, 'top_features_file') and self.top_features_file:
            if not os.path.exists(self.top_features_file):
                raise FileNotFoundError(f"Top-feature ranking file not found: {self.top_features_file}")
            allowed_features = read_top_features(self.top_features_file, self.top_features_k)

        # load dataset
        ds = Dataset(datafile, allowed_features=allowed_features)
        if not hasattr(self, 'drug_n_oversample'):
            self.drug_n_oversample = 1
        # add examples to trainer
        for xseq, yseq, _ in ds.instances() :
            # Oversample sequences with minority labels to increase their training impact.
            repeats = self.drug_n_oversample if ('B-drug_n' in yseq or 'I-drug_n' in yseq) else 1
            for _ in range(repeats):
                self.trainer.append(xseq, yseq, 0)

        # train and store model 
        self.trainer.train(self.modelfile, -1)

        
    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        if self.tagger is None :
            print("This model has not been trained", file=sys.stderr)
            sys.exit(1)

        return self.tagger.tag(xseq)

