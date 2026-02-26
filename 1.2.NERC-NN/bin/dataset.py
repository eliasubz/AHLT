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
            self.data = {}
            # parse XML file, obtaining a DOM tree
            tree = parse(filename)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences :
                sid = s.attributes["id"].value   # get sentence id
                stext = s.attributes["text"].value   # get sentence text
                print(f"parsing sentence {sid}        \r", end="")
                entities = s.getElementsByTagName("entity")
                spans = []
                for e in entities :
                    # for discontinuous entities, we only get the first span
                    # (will not work, but there are few of them)
                    (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
                    typ =  e.attributes["type"].value
                    spans.append((int(start),int(end),typ))

                # convert the sentence to a list of tokens
                tokens = nlp(stext)
                # add gold label to each token, and store it in self.data
                self.data[sid] = {'stext': stext, 'tokens': tokens, 'labels': []}
                for tk in tokens :
                    # see if the token is part of an entity
                    tks,tke = tk.idx, tk.idx+len(tk.text)
                    # store gold standard tag for this token
                    self.data[sid]['labels'].append(self.__get_label(tks,tke,spans))

 
    ## --------- get label ----------- 
    ##  Find out whether given token is marked as part of an entity in the XML
    def __get_label(self, tks, tke, spans) :
        for (spanS,spanE,spanT) in spans :
            if tks==spanS and tke<=spanE+1 : return "B-"+spanT
            elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
        return "O"
 
    ## ---- iterator to all get sentences in the data set
    def sentences(self) :
        for sid in self.data :
            yield self.data[sid]['stext'], self.data[sid]['tokens'], self.data[sid]['labels']

    ## ---- iterator to get ids for sentence in the data set
    def sentence_ids(self) :
        for sid in self.data :
            yield sid

    ## ---- get sentence by id
    def get_sentence_tokens(self, sid) :
        return self.data[sid]['tokens']

    ## ---- get sentence by id
    def get_sentence_labels(self, sid) :        
        return self.data[sid]['labels']
        
    ## ---- get sentence by id
    def get_sentence_text(self, sid) :        
        return self.data[sid]['stext']

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

