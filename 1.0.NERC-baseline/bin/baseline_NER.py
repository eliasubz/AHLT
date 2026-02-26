#! /usr/bin/python3

import os, sys
import json
from xml.dom.minidom import parse
import spacy

from drug_index import *

## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def extract_entities(stext, tokens, index) :
    result = []
    i = 0
    while i < len(tokens) :
        # check if a drug name starts at position i
        drug_type, end = index.find_drug(tokens, i)

        if drug_type is not None :
            entity_start = tokens[i].idx
            entity_end = tokens[end].idx + len(tokens[end].text)
            e = { "offset" : str(entity_start)+"-"+str(entity_end-1),
                  "text" : stext[entity_start:entity_end],
                  "type" : drug_type
                 }
            result.append(e)
            i = end
        
        i += 1

    return result
      
## --------- Entity extractor baseline ----------- 
def NER_baseline(datafile, drugindex, outfile) :
    outf = open(outfile, "w")
    
    index = DrugIndex(drugindex)

    # create tokenizer
    nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])

    # parse XML file, obtaining a DOM tree
    tree = parse(datafile)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        print(f"processing sentence {sid}        \r", end="")

        # tokenize text with spacy tokenizer
        tokens = nlp(stext)
        # extract entities in text
        entities = extract_entities(stext, tokens, index)

        # print sentence entities in format requested for evaluation
        for e in entities :
            print(sid,
                  e["offset"],
                  e["text"],
                  e["type"],
                  sep = "|",
                  file = outf)

    outf.close()


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir drug_index
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

if __name__ == "__main__" :
   if len(sys.argv) != 4 :
       print(f"usage:  {os.path.basename(__file__)} datafile drug_index  result.out")
       sys.exit(0)

   datafile = sys.argv[1]
   drugidx = sys.argv[2]
   outfile = sys.argv[3]

   # load previously created index
   NER_baseline(datafile, drugidx, outfile)












