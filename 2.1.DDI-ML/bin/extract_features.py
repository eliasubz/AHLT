#! /usr/bin/python3

import sys, os
from xml.dom.minidom import parse
import spacy

from patterns import *

## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_pair_features(tree, entities, e1, e2) :
   feats = set()

   # Features about entity types
   feats.add("typeE1="+ entities[e1]['type'])
   feats.add("typeE2="+ entities[e2]['type'])
   if entities[e1]['text'].lower() == entities[e2]['text'].lower() : 
      feats.add("samedrug")

   # features about paths in the tree.
   # get head token for each gold entity
   tkE1 = get_fragment_head(tree,entities[e1]['start'],entities[e1]['end'])
   tkE2 = get_fragment_head(tree,entities[e2]['start'],entities[e2]['end'])
   if tkE1 is not None and tkE2 is not None:      
      # get LCS      
      lcs = get_LCS(tree,tkE1,tkE2)

      if lcs is not None :
          feats.add("lcs="+lcs.lemma_+"_"+lcs.pos_)
          
          # paths from E1 to LCS, using lemma, rel, or both
          path1 = get_up_path(tkE1,lcs)
          p1 = "<".join([x.lemma_+"_"+x.dep_ for x in path1])
          feats.add("path1="+p1)
          p1b = "<".join([x.lemma_ for x in path1])
          feats.add("path1b="+p1b)
          p1c = "<".join([x.dep_ for x in path1])
          feats.add("path1c="+p1c)

          # paths from LCS to E2, using lemma, rel, or both
          path2 = get_down_path(lcs,tkE2)
          p2 = ">".join([x.lemma_+"_"+x.dep_ for x in path2])
          feats.add("path2="+p2)
          p2b = ">".join([x.lemma_ for x in path2])
          feats.add("path2b="+p2b)
          p2c = ">".join([x.dep_ for x in path2])
          feats.add("path2c="+p2c)

          # paths from E1 to E2, using lemma, rel, or both
          p = p1+"<"+lcs.lemma_+"_"+lcs.dep_+">"+p2
          feats.add("path="+p)
          pb = p1b+"<"+lcs.lemma_+">"+p2b
          feats.add("pathb="+pb)
          pc = p1c+"<"+lcs.dep_+">"+p2c
          feats.add("pathc="+pc)

          # LCS lemma/tag and rels under it
          if len(path1)>0 and len(path2)>0 :
             pa = path1[-1].dep_+"<"+lcs.lemma_+">"+path2[0].dep_
             feats.add("pathA="+pa)
             pab = path1[-1].dep_+"<"+lcs.pos_+">"+path2[0].dep_
             feats.add("pathAb="+pab)

          # words in path from E1 to E2
          for w in path1 :
             feats.add("wip1="+w.lemma_)
             feats.add("wip="+w.lemma_)
          for w in path2 :
             feats.add("wip2="+w.lemma_)
             feats.add("wip="+w.lemma_)
          feats.add("wip="+lcs.lemma_)
          feats.add("lcs="+lcs.lemma_)

          # lcs children
          for w in lcs.children : feats.add("lcsCH="+w.lemma_)
      
   # features using rule-based patterns
   for pat in patterns :
      match = patterns[pat](tree, entities, e1, e2)
      if match is not None: 
         for m in match :
            feats.add(pat+"="+m)
                     
   return feats


## --------- Feature extractor ----------- 
## -- Extract features for each entity pair in each
## -- sentence in given file

def extract_features(datafile, outfile, dump_trees=False) :

   # open output file
   outf = open(outfile, "w")
   if dump_trees:
       treedir = os.path.join(os.path.dirname(outfile), "svg")
       os.makedirs(treedir, exist_ok=True)
    
   # create spacy parser
   nlp = spacy.load("en_core_web_trf",
                    enable=["transformer", "tagger","attribute_ruler", "lemmatizer", "ner", "parser"])

   # parse XML file, obtaining a DOM tree
   tree = parse(datafile)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        print(f"extracting sentence {sid}             \r", end="")
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           text = e.attributes["text"].value
           typ = e.attributes["type"].value
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1]),
                           'text': text, 'type' : typ}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # get syntactic analysis for the sentence
        analysis = nlp(stext)
        if dump_trees : 
           svg = spacy.displacy.render(analysis,style="dep")    
           with open(os.path.join(treedir,sid+".svg"),"w") as sf :  
              sf.write(svg)       
        
        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction
            feats = extract_pair_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t", file=outf)


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
    # directory with files to process
    datafile = sys.argv[1]
    # file where to store results
    featfile = sys.argv[2]
    trees = len(sys.argv)>3 and sys.argv[3]=="trees"
    
    extract_features(datafile, featfile, trees)

