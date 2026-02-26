#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries

## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans) :
    for (spanS,spanE,spanT) in spans :
        if tks==spanS and tke<=spanE+1 : return "B-"+spanT
        elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
    return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_sentence_features(tokens, dicts) :

   # for each token, generate list of features and add it to the result
   sentenceFeatures = {}
   for i,tk in enumerate(tokens) :
      tokenFeatures = []
      t = tk.text

      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      if t.isupper() : tokenFeatures.append("isUpper")
      if t.istitle() : tokenFeatures.append("isTitle")
      if t.isdigit() : tokenFeatures.append("isDigit")
      if '-' in t : tokenFeatures.append("hasDash")
      if re.search('[0-9]',t) : tokenFeatures.append("hasDigit")
      found,val = dicts.find(t.lower(), 'external')
      if found:
         for c in val : tokenFeatures.append("external="+c)
      found,val = dicts.find(t.lower(), 'externalpart')
      if found:
          for c in val : tokenFeatures.append("externalpart="+c)

      if i>0 :
         tPrev = tokens[i-1].text
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         if tPrev.isupper() : tokenFeatures.append("isUpperPrev")
         if tPrev.istitle() : tokenFeatures.append("isTitlePrev")
         if tPrev.isdigit() : tokenFeatures.append("isDigitPrev")
         if '-' in tPrev : tokenFeatures.append("hasDashPrev")
         if re.search('[0-9]',tPrev) : tokenFeatures.append("hasDigitPrev")
         found,val = dicts.find(tPrev.lower(), 'external')
         if found:
             for c in val : tokenFeatures.append("externalPrev="+c)
         found,val = dicts.find(tPrev.lower(), 'externalpart')
         if found:
             for c in val : tokenFeatures.append("externalpartPrev="+c)
      else :
         tokenFeatures.append("BoS")

      if i<len(tokens)-1 :
         tNext = tokens[i+1].text
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         if tNext.isupper() : tokenFeatures.append("isUpperNext")
         if tNext.istitle() : tokenFeatures.append("isTitleNext")
         if tNext.isdigit() : tokenFeatures.append("isDigitNext")
         if '-' in tNext : tokenFeatures.append("hasDashNext")
         if re.search('[0-9]',tNext) : tokenFeatures.append("hasDigitNext")
         found,val = dicts.find(tNext.lower(), 'external')
         if found:
            for c in val : tokenFeatures.append("externalNext="+c)
         found,val = dicts.find(tNext.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartNext="+c)
      else:
         tokenFeatures.append("EoS")
    
      sentenceFeatures[i] = tokenFeatures
    
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile) :

    # load dictionaries
    dicts = Dictionaries(os.path.join(paths.RESOURCES,"dictionaries.json"))

    # open output file
    outf = open(outfile, "w")
    
    # create analyzer. We don't need the parser now, it will be faster if disabled
    nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])
    
    # parse XML file, obtaining a DOM tree
    tree = parse(datafile)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      print(f"extracting sentence {sid}        \r", end="")
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity") # get gold standard entities
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))

      # convert the sentence to a list of tokens
      tokens = nlp(stext)
      # extract sentence features
      features = extract_sentence_features(tokens, dicts)

      # print features in format expected by CRF/SVM/MEM trainers
      for i,tk in enumerate(tokens) :
         # see if the token is part of an entity
         tks,tke = tk.idx, tk.idx+len(tk.text)
         # get gold standard tag for this token
         tag = get_label(tks, tke, spans)
         # print feature vector for this token
         print (sid, tk.text, tks, tke-1, tag, "\t".join(features[i]), sep='\t', file=outf)

      # blank line to separate sentences
      print(file=outf)

    # close output file
    outf.close()

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
    
    extract_features(datafile, featfile)

