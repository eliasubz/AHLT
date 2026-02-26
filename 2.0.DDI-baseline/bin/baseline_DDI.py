#! /usr/bin/python3

import sys
import re
import os
from xml.dom.minidom import parse
import spacy

wib = { "effect" : ["alcohol", "response", "enhance", "action",
                    "central", "additive", "nervous", "man", "block",
                    "antagonize", "bleed", "weakness", "hyperreflexia",
                    "incoordination", "rarely", "prothrom", "NIMBEX",
                    "adrenocortical", "secondary"],
        "int" : ["interact", "tylenol", "mivacron"],
        "mechanism" : ["due", "tubular", "cyp", "induction", "delay", 
                       "acetazolamide", "acute", "displace", "likely", 
                       "cyp3a", "modest", "gastric", "ability", "media", 
                       "liver", "anticipate"],
        "advise" : ["dihydroergotamine", "cyp2d6", "isoenzyme", "solution",
                    "isozyme", "narrow", "pediatric", "buprenorphine", 
                    "extensively", "start", "tobramycin", "possibility", 
                    "ordinarily", "methysergide", "tell", "doctor", "index",
                    "vinblastine", "cautiously", "adjust"]
       }

inverse_wib = {w:t for t in wib for w in wib[t]}

## -------------------------------------------------
## check if given sentence expresses an interaction bewtween e1 and e2
def check_interaction(tokens, entities, e1, e2) :
   
   for tk in tokens :
      if entities[e1]['end'] < tk.idx and tk.idx+len(tk.text) < entities[e2]['start']:
         # the token is between the entities, check its form
         if tk.text in inverse_wib :
            return inverse_wib[tk.lemma_]
         
   return None
         
   
## --------- DDI extractor baseline ----------- 
def DDI_baseline(datafile, outfile) :
   outf = open(outfile, "w")
   
   # create tokenizer
   nlp = spacy.load("en_core_web_trf", enable=["tokenizer","tagger","attribute_ruler","lemmatizer"])
   
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
        
      # load sentence entities
      entities = {}
      ents = s.getElementsByTagName("entity")
      for e in ents :
         id = e.attributes["id"].value
         offs = e.attributes["charOffset"].value.split("-")           
         entities[id] = {'start': int(offs[0]),
                         'end': int(offs[-1]),
                         'type': e.attributes["type"].value,
                         'text': e.attributes["text"].value}
           
      # for each pair in the sentence, decide whether it is DDI and its type
      pairs = s.getElementsByTagName("pair")      
      for p in pairs:
         id_e1 = p.attributes["e1"].value
         id_e2 = p.attributes["e2"].value
           
         ddi_type = check_interaction(tokens, entities, id_e1, id_e2)
         if ddi_type is not None :
            print("|".join([sid, id_e1, id_e2, ddi_type]), file=outf)
           
   outf.close()
        
        
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir drug_index
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

if __name__ == "__main__" :
   if len(sys.argv) != 3 :
      print(f"usage:  {os.path.basename(__file__)} datafile result.out")
      sys.exit(0)
      
   datafile = sys.argv[1]
   outfile = sys.argv[2]
      
   # load previously created index
   DDI_baseline(datafile, outfile)
      
        
        
        
        
        
