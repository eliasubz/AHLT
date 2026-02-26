#####################################################
## Class to store an ngram ME model
#####################################################

import sys
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
            maxit =  int(params['max_iterations']) if 'max_iterations' in params else 9999999
            c1 = float(params['c1']) if 'c1' in params else 0.1
            c2 = float(params['c2']) if 'c2' in params else 1.0
            eps = float(params['epsilon']) if 'epsilon' in params else 0.00001
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
        # load dataset
        ds = Dataset(datafile)
        # add examples to trainer
        for xseq, yseq, _ in ds.instances() :
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

