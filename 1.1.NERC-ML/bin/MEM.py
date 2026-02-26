#####################################################
## Class to store an ngram ME model
#####################################################
import sys
import pickle

import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

import dataset


class MEM:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile, params=None):

        self.modelfile = modelfile
        if params is None:
            # only modelfile given, assume it is an existing model and load it        
            with open(self.modelfile, 'rb') as df :
                self.tagger = pickle.load(df)
            with open(self.modelfile+".idx", 'rb') as df :
                self.fidx = pickle.load(df)
                
        else :  # params given, create new empty model

            # extract parameters if provided. Use default if not
            C = float(params['C']) if 'C' in params else 1.0
            solver = params['solver'] if 'solver' in params else 'lbfgs'
            maxit = params['max_iter'] if 'max_iter' in params else 1500

            # create and train empty classifier with given parameters
            self.tagger = LogisticRegression(verbose=1,
                                             C=C,
                                             solver=solver,
                                             max_iter=maxit)

                
    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile):
        # load dataset
        ds = dataset.Dataset(datafile)
        self.fidx = ds.feature_index()

        # Read training instances 
        X,Y = ds.csr_matrix()

        # train classifier
        self.tagger.fit(X,Y)

        # save model
        pickle.dump(self.tagger, open(self.modelfile, 'wb'))
        pickle.dump(self.fidx, open(self.modelfile+".idx", 'wb'))
    

    ## --------------------------------------------------
    ## predict best class for given example
    ## --------------------------------------------------
    def predict(self, xseq):

        if len(xseq)==0 : return []
        
        # Encode xseq into a CSR sparse matrix
        rowi = [] # row (example number)
        colj = [] # column (feature number)
        data = [] # value (1 or 0 since we use binary features)
        nex = 0 # example  counter (each word is one example)
        for w in xseq :
            for f in w :
                if f in self.fidx :
                    data.append(1)
                    rowi.append(nex)
                    colj.append(self.fidx[f]) 
                    # next word           
            nex += 1
        X = scipy.sparse.csr_matrix((data, (rowi, colj)), shape=(len(xseq),len(self.fidx)))
        
        # apply model to X and return predictions
        return self.tagger.predict(X)


