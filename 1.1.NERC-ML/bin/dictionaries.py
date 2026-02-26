
import sys, os
import json
import spacy

import paths


# -------------------------------------------------------------------
class Dictionaries() :
    
  def __init__(self, filename=None) :
      
      if filename is not None :
          if filename.endswith(".json") :
              # parameter is a pickle file, load it
              with open(filename) as pf:
                  self.data = json.load(pf)
          else :
              print("ERROR. Expected .json file", file = sys.stderr)
              sys.exit(1)
        
      else :
          # create stanza tokenizer
          nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])

          self.data = {}
          self.data['external'] = {}
          self.data['externalpart'] = {}
          print("Processing HSDB")
          with open(os.path.join(paths.RESOURCES,"HSDB.txt")) as h :
              for x in h.readlines() :
                  self.data['external'][x.strip().lower()] = set(["drug"])
                  wds = nlp(x)
                  if len(wds)>1 : 
                      for tk in wds :
                          self.data['externalpart'][tk.text] = set(["drug"])

          print("Processing DrugBank")
          with open(os.path.join(paths.RESOURCES,"DrugBank.txt")) as h :
              for x in h.readlines() :
                  (n,t) = x.strip().lower().split("|")
                  if n in self.data['external'] : self.data['external'][n].add(t)
                  else : self.data['external'][n] = set([t])
                  wds = nlp(n)
                  if len(wds)>1 : 
                      for tk in wds :
                          if tk.text in self.data['externalpart'] : self.data['externalpart'][tk.text].add(t)
                          else : self.data['externalpart'][tk.text] = set([t])

          print("Processing train")
          with open(os.path.join(paths.RESOURCES,"drugs-train.txt")) as h :
              for x in h.readlines() :
                  (_,_,n,t) = x.strip().lower().split("|")
                  if n in self.data['external'] : self.data['external'][n].add(t)
                  else : self.data['external'][n] = set([t])
                  wds = nlp(n)
                  if len(wds)>1 : 
                      for tk in wds :
                          if tk.text in self.data['externalpart'] : self.data['externalpart'][tk.text].add(t)
                          else : self.data['externalpart'][tk.text] = set([t])

  ## ---- find entry in dictionaries
  def find(self, txt, section) :
    if txt in self.data[section] :
      return True, self.data[section][txt]
    else :
      return False, None

  ## ---- save dataset to json file
  def save(self, filename) :
      for x in self.data:
         for y in self.data[x] :
             self.data[x][y] = list(self.data[x][y])
             
      with open(filename+".json", "w") as pf:
          json.dump(self.data, pf, indent=3, ensure_ascii=False)

