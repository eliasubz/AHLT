import os, sys

HERE = os.path.abspath(os.path.dirname(__file__)) # location of this file

# one level up, current classifier approach for this task
CLASSIFIER = os.path.dirname(HERE) 
# needed directories for this classifier 
PREPROCESS = os.path.join(CLASSIFIER,"preprocessed")
MODELS = os.path.join(CLASSIFIER,"models")
RESULTS = os.path.join(CLASSIFIER,"results")

# three levels up, main project dir
MAIN = os.path.dirname(os.path.dirname(os.path.dirname(HERE))) 
# useful project directories
DATA = os.path.join(MAIN,"data") # down to "data"
RESOURCES = os.path.join(MAIN,"resources") # down to "resources"
UTIL = os.path.join(MAIN,"util") # down to "util"
# some useful scripts there
sys.path.append(UTIL)

