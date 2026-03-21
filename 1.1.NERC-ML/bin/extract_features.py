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
 
## --------- Helper functions -----------

ROMAAN = re.compile(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.IGNORECASE)

# Common drug name suffixes (pharmacological morphology)
DRUG_SUFFIX = re.compile(
   r'(mab|nib|pril|olol|cillin|mycin|statin|oxacin|azole|dipine|sartan'
   r'|tidine|prazole|setron|vir|mide|amine|ine|ase|ide|ate|ium|ol)$', re.IGNORECASE)

GREEK = re.compile(r'alpha|beta|gamma|delta|epsilon|omega|kappa|lambda|sigma|theta', re.IGNORECASE)

def word_shape(t):
   """Map chars to X (upper), x (lower), d (digit), keep punctuation"""
   shape = ''
   for c in t:
      if c.isupper(): shape += 'X'
      elif c.islower(): shape += 'x'
      elif c.isdigit(): shape += 'd'
      else: shape += c
   return shape[:8]  # cap length to avoid feature explosion

def collapsed_shape(t):
   """Word shape with runs of same char collapsed: Xxxxx -> Xx, XXXX -> X"""
   raw = word_shape(t)
   if not raw: return raw
   result = raw[0]
   for c in raw[1:]:
      if c != result[-1]: result += c
   return result

def token_length_bucket(t):
   l = len(t)
   if l <= 2: return 'len_short'
   elif l <= 5: return 'len_medium'
   elif l <= 10: return 'len_long'
   else: return 'len_verylong'

## -- Extract features for each token in given sentence

def extract_sentence_features(tokens, dicts) :

   # for each token, generate list of features and add it to the result
   sentenceFeatures = {}
   for i,tk in enumerate(tokens) :
      tokenFeatures = []
      t = tk.text

      # --- current token: original features ---
      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      tokenFeatures.append("pref1="+t[:1])
      tokenFeatures.append("pref2="+t[:2])
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      if len(t)>0 and t[0].isupper() : tokenFeatures.append("isCapitalized")
      if t.isupper() : tokenFeatures.append("isUpper")
      if t.istitle() : tokenFeatures.append("isTitle")
      if t.isdigit() : tokenFeatures.append("isDigit")
      if '-' in t : tokenFeatures.append("hasDash")
      if re.search('[0-9]',t) : tokenFeatures.append("hasDigit")
      found,val = dicts.find(t.lower(), 'external')
      hasExternalDrug = False
      if found:
         for c in val :
            tokenFeatures.append("external="+c)
            if c in ('drug','brand','group') : hasExternalDrug = True
      found,val = dicts.find(t.lower(), 'externalpart')
      hasExternalpartDrugN = False
      if found:
          for c in val :
             tokenFeatures.append("externalpart="+c)
             if c == 'drug_n' : hasExternalpartDrugN = True
      # signal: token is not in any standard drug database (potential novel/experimental drug)
      if not hasExternalDrug : tokenFeatures.append("notInDrugDB")
      if hasExternalpartDrugN and not hasExternalDrug : tokenFeatures.append("likelyDrugN")
      # --- current token: new features ---
      tokenFeatures.append("suf1="+t[-1:])
      tokenFeatures.append("suf2="+t[-2:])
      tokenFeatures.append("pref3="+t[:3])
      tokenFeatures.append("pref4="+t[:4])
      tokenFeatures.append("shape="+word_shape(t))
      tokenFeatures.append("cshape="+collapsed_shape(t))
      tokenFeatures.append(token_length_bucket(t))
      if re.search('[A-Z]', t[1:]) : tokenFeatures.append("hasInternalUpper")
      if re.search('[()\[\]]', t) : tokenFeatures.append("hasBracket")
      if '/' in t : tokenFeatures.append("hasSlash")
      if '.' in t : tokenFeatures.append("hasDot")
      if t.isupper() and len(t) <= 5 : tokenFeatures.append("isShortUpper")
      if ROMAAN.match(t) and len(t) > 0 : tokenFeatures.append("isRoman")
      if DRUG_SUFFIX.search(t) : tokenFeatures.append("hasDrugSuffix")
      if GREEK.search(t) : tokenFeatures.append("hasGreek")

      # --- previous token (i-1) ---
      if i>0 :
         tPrev = tokens[i-1].text
         # original features
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("pref1Prev="+tPrev[:1])
         tokenFeatures.append("pref2Prev="+tPrev[:2])
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         if len(tPrev)>0 and tPrev[0].isupper() : tokenFeatures.append("isCapitalizedPrev")
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
         # new features
         tokenFeatures.append("suf1Prev="+tPrev[-1:])
         tokenFeatures.append("suf2Prev="+tPrev[-2:])
         tokenFeatures.append("pref3Prev="+tPrev[:3])
         tokenFeatures.append("pref4Prev="+tPrev[:4])
         tokenFeatures.append("shapePrev="+word_shape(tPrev))
         tokenFeatures.append("cshapePrev="+collapsed_shape(tPrev))
         tokenFeatures.append(token_length_bucket(tPrev)+"Prev")
         if re.search('[A-Z]', tPrev[1:]) : tokenFeatures.append("hasInternalUpperPrev")
         if re.search('[()\.\[\]]', tPrev) : tokenFeatures.append("hasBracketPrev")
         if tPrev.isupper() and len(tPrev) <= 5 : tokenFeatures.append("isShortUpperPrev")
         if ROMAAN.match(tPrev) and len(tPrev) > 0 : tokenFeatures.append("isRomanPrev")
         if DRUG_SUFFIX.search(tPrev) : tokenFeatures.append("hasDrugSuffixPrev")
         if GREEK.search(tPrev) : tokenFeatures.append("hasGreekPrev")
      else :
         tokenFeatures.append("BoS")

      # --- token two positions back (i-2): new context window ---
      if i > 1 :
         t2 = tokens[i-2].text
         tokenFeatures.append("formPrev2="+t2)
         tokenFeatures.append("formlowerPrev2="+t2.lower())
         tokenFeatures.append("suf3Prev2="+t2[-3:])
         tokenFeatures.append("shapePrev2="+word_shape(t2))
         if t2.isupper() : tokenFeatures.append("isUpperPrev2")
         if re.search('[0-9]', t2) : tokenFeatures.append("hasDigitPrev2")
         found,val = dicts.find(t2.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartPrev2="+c)
      else :
         tokenFeatures.append("BoS2")

      # --- next token (i+1) ---
      if i<len(tokens)-1 :
         tNext = tokens[i+1].text
         # original features
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("pref1Next="+tNext[:1])
         tokenFeatures.append("pref2Next="+tNext[:2])
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         if len(tNext)>0 and tNext[0].isupper() : tokenFeatures.append("isCapitalizedNext")
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
         # new features
         tokenFeatures.append("suf1Next="+tNext[-1:])
         tokenFeatures.append("suf2Next="+tNext[-2:])
         tokenFeatures.append("pref3Next="+tNext[:3])
         tokenFeatures.append("pref4Next="+tNext[:4])
         tokenFeatures.append("shapeNext="+word_shape(tNext))
         tokenFeatures.append("cshapeNext="+collapsed_shape(tNext))
         tokenFeatures.append(token_length_bucket(tNext)+"Next")
         if re.search('[A-Z]', tNext[1:]) : tokenFeatures.append("hasInternalUpperNext")
         if re.search('[()\.\[\]]', tNext) : tokenFeatures.append("hasBracketNext")
         if tNext.isupper() and len(tNext) <= 5 : tokenFeatures.append("isShortUpperNext")
         if ROMAAN.match(tNext) and len(tNext) > 0 : tokenFeatures.append("isRomanNext")
         if DRUG_SUFFIX.search(tNext) : tokenFeatures.append("hasDrugSuffixNext")
         if GREEK.search(tNext) : tokenFeatures.append("hasGreekNext")
         # bigram: current + next (joint context feature)
         tokenFeatures.append("bigram="+t.lower()+"_"+tNext.lower())
         # other combinations
         tokenFeatures.append("shapePair="+word_shape(t)+"_"+word_shape(tNext))
      else:
         tokenFeatures.append("EoS")

      # current and previous words together
      if i>0 :
         tokenFeatures.append("bigramPrev="+tokens[i-1].text.lower()+"_"+t.lower())
      # previous, current, and next words together
      if i>0 and i<len(tokens)-1 :
         tokenFeatures.append("trigramPrevCurrNext="+tokens[i-1].text.lower()+"_"+t.lower()+"_"+tokens[i+1].text.lower())

      # --- token two positions ahead (i+2): new context window ---
      if i < len(tokens)-2 :
         t2 = tokens[i+2].text
         tokenFeatures.append("formNext2="+t2)
         tokenFeatures.append("formlowerNext2="+t2.lower())
         tokenFeatures.append("suf3Next2="+t2[-3:])
         tokenFeatures.append("shapeNext2="+word_shape(t2))
         if t2.isupper() : tokenFeatures.append("isUpperNext2")
         if re.search('[0-9]', t2) : tokenFeatures.append("hasDigitNext2")
         found,val = dicts.find(t2.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartNext2="+c)
      else:
         tokenFeatures.append("EoS2")

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

      # enable NER so we can add known-person/known-location features
   nlp = spacy.load("en_core_web_trf", enable=["tokenizer", "transformer", "tagger", "attribute_ruler", "lemmatizer", "ner"])

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

