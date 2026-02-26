import os, sys
from xml.dom.minidom import parse
import pickle
import torch
import spacy

class Dataset:
    ##  Parse all XML files in given dir, and load a list of sentences.
    ##  Each sentence is a list of tuples (word, start, end, tag)
    def __init__(self, filename) :
        if filename[-4:] == ".pck" :
            # parameter is a pickle file, load it
            with open(filename, "rb") as pf:
                self.data = pickle.load(pf)

        else : # parameter should be an XML file, load it
            # create spacy Pos Tagger & lemmatizer
            if torch.cuda.is_available() : spacy.require_gpu()
            nlp = spacy.load("en_core_web_trf")
            self.data = []
            # parse XML file, obtaining a DOM tree
            tree = parse(filename)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences :
                sid = s.attributes["id"].value   # get sentence id
                stext = s.attributes["text"].value   # get sentence text
                print(f"parsing sentence {sid}        \r", end="")
                
                ents = s.getElementsByTagName("entity")
                if len(ents) <= 1 : continue
                
                entities = {}
                for e in ents :
                    # for discontinuous entities, we only get the first span
                    # (will not work, but there are few of them)
                    eid =  e.attributes["id"].value
                    typ =  e.attributes["type"].value
                    (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
                    entities[eid] = {"start":int(start), "end":int(end), "type": typ}

                # convert the sentence to a list of tokens
                tokens = nlp(stext)                
                # for each pair in the sentence, get whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    # ground truth
                    ddi = p.attributes["ddi"].value
                    if (ddi=="true") : dditype = p.attributes["type"].value
                    else : dditype = "null"
                    # target entities
                    e1 = p.attributes["e1"].value
                    e2 = p.attributes["e2"].value
                
                    sent = []
                    seen = set([])
                    for token in tokens :
                        if token.text.startswith(" "): continue
                        tk_ent = self.__is_entity(token, entities)
                        
                        if tk_ent is None : 
                           token = {'form': token.text, 
                                    'lc_form': token.text.lower(), 
                                    'lemma': token.lemma_,
                                    'pos': token.pos_}
                        elif tk_ent == e1 : 
                            token = {'form':'<DRUG1>', 
                                     'lc_form':'<DRUG1>', 
                                     'lemma':'<DRUG1>', 
                                     'pos':'<DRUG1>', 
                                     'etype':entities[e1]['type']}
                        elif tk_ent == e2 : 
                            token = {'form':'<DRUG2>', 
                                     'lc_form':'<DRUG2>', 
                                     'lemma':'<DRUG2>', 
                                     'pos':'<DRUG2>', 
                                     'etype':entities[e2]['type']}
                        else :       
                            token = {'form':'<DRUG_OTHER>', 
                                     'lc_form':'<DRUG_OTHER>', 
                                     'lemma':'<DRUG_OTHER>', 
                                     'pos':'<DRUG_OTHER>', 
                                     'etype':entities[tk_ent]['type']}                        
                    
                        if tk_ent==None or tk_ent not in seen : sent.append(token)
                        if tk_ent!=None : seen.add(tk_ent)
                    
                    # resulting vector
                    self.data.append({'sid': sid, 'e1':e1, 'e2':e2, 'type':dditype, 'sent':sent})

 
    ## --------------------------------------------------------------
    ## check whether a token belongs to one of given entities
    def __is_entity(self, tk, entities):
        for e in entities :
            if entities[e]["start"] <= tk.idx and tk.idx+len(tk.text) <= entities[e]["end"]+1 :
                return e
        return None    
         
    ## ---- iterator to all get sentences in the data set
    def sentences(self) :
        for s in self.data :
            yield s
            
    ## ---- save dataset to pickle file
    def save(self, filename) :
        if not filename.endswith(".pck") : filename += ".pck"
        with open(filename, "wb") as pf:
            pickle.dump(self.data, pf)
  
## ------------- MAIN ----------------
## parse given XML file, store results in 
## pickle file for later use

if __name__ == "__main__" :
    datafile = sys.argv[1]
    picklfile =  sys.argv[2]
    data = Dataset(datafile)
    data.save(filename)

