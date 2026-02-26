#! /usr/bin/python3

import sys, os
from baseline_DDI import DDI_baseline

import paths
sys.path.append(paths.UTIL)
from evaluator import evaluate

os.makedirs(paths.RESULTS, exist_ok=True)
for ds in ["devel", "test"]:
   print(f"Running baseline on {ds}                   ")
   DDI_baseline(os.path.join(paths.DATA,f"{ds}.xml"), 
                os.path.join(paths.RESULTS,f"{ds}.out"))
   print(f"Evaluating baseline on {ds}                ")
   evaluate("DDI",
            os.path.join(paths.DATA,f"{ds}.xml"),
            os.path.join(paths.RESULTS,f"{ds}.out"),
            os.path.join(paths.RESULTS,f"{ds}.stats"))

