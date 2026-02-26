#! /usr/bin/python3

import sys, os
from drug_index import DrugIndex
from baseline_NER import NER_baseline

import paths
from gold_extractor import GoldExtractor
from evaluator import evaluate

# if feature extraction is required, do it
print("Extracting drugs from train data")
gold = GoldExtractor(os.path.join(paths.DATA,"train.xml"))
gold.extract_NER(os.path.join(paths.RESOURCES,"drugs-train.txt"))
print("Creating prefix-tree index with all known drug names")
idx = DrugIndex(resources=paths.RESOURCES)    
idxfile = os.path.join(paths.RESOURCES,"drug-index.json")
with open(idxfile,"w") as jf: idx.dump(file=jf)

print("Applying index to predict drugs")
os.makedirs(paths.RESULTS, exist_ok=True)
for ds in ["devel", "test"]:
   print(f"Running baseline on {ds}                   ")
   NER_baseline(os.path.join(paths.DATA,f"{ds}.xml"), 
                idxfile, 
                os.path.join(paths.RESULTS,f"{ds}.out"))
   print(f"Evaluating baseline on {ds}                ")
   evaluate("NER",
            os.path.join(paths.DATA,f"{ds}.xml"),
            os.path.join(paths.RESULTS,f"{ds}.out"),
            os.path.join(paths.RESULTS,f"{ds}.stats"))

