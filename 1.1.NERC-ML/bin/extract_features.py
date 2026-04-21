#! /usr/bin/python3

import sys, os
import re
from dataclasses import dataclass
from typing import Optional, Set
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries


@dataclass(frozen=True)
class FeatureFlags:
    """Optional feature groups for ablation. All True = current full extractor."""
    med_patterns: bool = True
    stopwords: bool = True
    external_phrase: bool = True
    affix5_alpha: bool = True
    length_extras: bool = True
    length_next: bool = True


FEATURE_FLAG_KEYS = (
    "med_patterns",
    "stopwords",
    "external_phrase",
    "affix5_alpha",
    "length_extras",
    "length_next",
)
_KNOWN_FLAGS: Set[str] = set(FEATURE_FLAG_KEYS)


def parse_feature_flags_param(value: Optional[str]) -> FeatureFlags:
    """
    None -> all True (default / backward compatible).
    '' (empty string) -> all False (base ablation profile).
    'med_patterns,stopwords' -> only those True, others False.
    """
    if value is None:
        return FeatureFlags()
    v = value.strip()
    if v == "":
        return FeatureFlags(
            med_patterns=False,
            stopwords=False,
            external_phrase=False,
            affix5_alpha=False,
            length_extras=False,
            length_next=False,
        )
    enabled = {x.strip() for x in v.split(",") if x.strip()}
    unknown = enabled - _KNOWN_FLAGS
    if unknown:
        raise ValueError(f"Unknown feature_flags keys: {sorted(unknown)}. Valid: {sorted(_KNOWN_FLAGS)}")
    return FeatureFlags(
        med_patterns="med_patterns" in enabled,
        stopwords="stopwords" in enabled,
        external_phrase="external_phrase" in enabled,
        affix5_alpha="affix5_alpha" in enabled,
        length_extras="length_extras" in enabled,
        length_next="length_next" in enabled,
    )


def feature_flags_to_csv_tuple(flags: FeatureFlags) -> tuple:
    """Fixed column order for sweep_metrics.tsv."""
    return tuple(int(getattr(flags, k)) for k in FEATURE_FLAG_KEYS)

try:
   # Prefer NLTK stopwords if available in your environment.
   from nltk.corpus import stopwords  # type: ignore
   STOP_WORD_SET = set(stopwords.words("english"))
except Exception:
   # Fallback to spaCy built-in stop words (avoids needing an NLTK download).
   from spacy.lang.en.stop_words import STOP_WORDS  # type: ignore
   STOP_WORD_SET = set(STOP_WORDS)

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

# Medication context patterns (dose/route/frequency/formulation cues)
DOSE_NUMBER = re.compile(r'^\d+(\.\d+)?$')
DOSE_UNIT = re.compile(r'^(mg|g|mcg|μg|ug|kg|ml|l|iu|u|%|mmol|meq)$', re.IGNORECASE)
FREQ = re.compile(r'^(qd|bid|tid|qid|qhs|qod|prn|stat|hs|q\d+h)$', re.IGNORECASE)
ROUTE = re.compile(r'^(po|iv|im|sc|sq|sl|pr|pv)$', re.IGNORECASE)
RELEASE = re.compile(r'^(xr|sr|er|cr|la)$', re.IGNORECASE)

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

def offset_suffix(offset):
   if offset < 0:
      d = abs(offset)
      return "Prev" if d == 1 else "Prev" + str(d)
   d = abs(offset)
   return "Next" if d == 1 else "Next" + str(d)

def boundary_marker(offset):
   d = abs(offset)
   if offset < 0:
      return "BoS" if d == 1 else "BoS" + str(d)
   return "EoS" if d == 1 else "EoS" + str(d)

def add_token_features(
    tokenFeatures, target_token_text, current_token_text, offset, dicts, feature_flags: FeatureFlags
):
   t = target_token_text
   s = "" if offset == 0 else offset_suffix(offset)
   
   # Simple stopword check; strips common punctuation around the token.
   t_low = t.lower()
   t_low_stripped = t_low.strip(".,;:!?()[]{}\"'")
   is_stop = t_low_stripped in STOP_WORD_SET

   # form/formlower only for the current token; in context window positions
   # these duplicate the explicit bigram/trigram features and inflate feature space.
   if offset == 0:
      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t_low)
   tokenFeatures.append("pref1"+s+"="+t[:1])
   tokenFeatures.append("pref2"+s+"="+t[:2])
   tokenFeatures.append("suf3"+s+"="+t[-3:])
   tokenFeatures.append("suf4"+s+"="+t[-4:])
   if len(t)>0 and t[0].isupper() : tokenFeatures.append("isCapitalized"+s)
   if t.isupper() : tokenFeatures.append("isUpper"+s)
   if t.istitle() : tokenFeatures.append("isTitle"+s)
   if t.isdigit() : tokenFeatures.append("isDigit"+s)
   if '-' in t : tokenFeatures.append("hasDash"+s)
   if re.search('[0-9]',t) : tokenFeatures.append("hasDigit"+s)

   found,val = dicts.find(t_low, 'external')
   hasExternalDrug = False
   if found:
      for c in val:
         tokenFeatures.append("external"+s+"="+c)
         if c in ('drug','brand','group') :
            hasExternalDrug = True

   found,val = dicts.find(t_low, 'externalpart')
   hasExternalpartDrugN = False
   if found:
      for c in val:
         tokenFeatures.append("externalpart"+s+"="+c)
         if c == 'drug_n' :
            hasExternalpartDrugN = True

   if hasExternalpartDrugN and not hasExternalDrug : tokenFeatures.append("likelyDrugN"+s)

   tokenFeatures.append("suf1"+s+"="+t[-1:])
   tokenFeatures.append("suf2"+s+"="+t[-2:])
   tokenFeatures.append("pref3"+s+"="+t[:3])
   tokenFeatures.append("pref4"+s+"="+t[:4])
   if feature_flags.affix5_alpha:
      tokenFeatures.append("suf5"+s+"="+t[-5:])
      tokenFeatures.append("pref5"+s+"="+t[:5])
      if t.isalpha():
         tokenFeatures.append("isAlpha"+s)
   tokenFeatures.append("shape"+s+"="+word_shape(t))
   tokenFeatures.append("cshape"+s+"="+collapsed_shape(t))
   tokenFeatures.append(token_length_bucket(t)+s)
   if re.search('[A-Z]', t[1:]) : tokenFeatures.append("hasInternalUpper"+s)
   if re.search(r'[()\[\]]', t) : tokenFeatures.append("hasBracket"+s)
   if '/' in t : tokenFeatures.append("hasSlash"+s)
   if '.' in t : tokenFeatures.append("hasDot"+s)
   if t.isupper() and len(t) <= 5 : tokenFeatures.append("isShortUpper"+s)
   if ROMAAN.match(t) and len(t) > 0 : tokenFeatures.append("isRoman"+s)
   if DRUG_SUFFIX.search(t) : tokenFeatures.append("hasDrugSuffix"+s)
   if GREEK.search(t) : tokenFeatures.append("hasGreek"+s)

   # medication context cues (help CRF boundary + disambiguation)
   if feature_flags.med_patterns:
      if DOSE_NUMBER.match(t):
         tokenFeatures.append("isDoseNumber"+s)
      if DOSE_UNIT.match(t):
         tokenFeatures.append("isDoseUnit"+s)
      if FREQ.match(t):
         tokenFeatures.append("isFreq"+s)
      if ROUTE.match(t):
         tokenFeatures.append("isRoute"+s)
      if RELEASE.match(t):
         tokenFeatures.append("isRelease"+s)

   if feature_flags.stopwords and is_stop:
      tokenFeatures.append("isStopWord"+s)

   if feature_flags.length_extras:
      if any(c.islower() for c in t) and any(c.isupper() for c in t):
         tokenFeatures.append("lowerAndUpper"+s)
      tokenFeatures.append("length"+s+"="+str(len(t)))

## -- Extract features for each token in given sentence

def extract_sentence_features(
    tokens, dicts, context_window: int = 1, feature_flags: Optional[FeatureFlags] = None
):
   if feature_flags is None:
      feature_flags = FeatureFlags()

   # for each token, generate list of features and add it to the result
   sentenceFeatures = {}

   phrase_B = [None] * len(tokens)
   phrase_I = [None] * len(tokens)
   if feature_flags.external_phrase:
      # Precompute multi-token dictionary phrase matches (e.g., "sodium chloride").
      low = [tk.text.lower() for tk in tokens]
      max_phrase_len = 4
      for i in range(len(tokens)):
         best = None  # (n, categories)
         for n in range(min(max_phrase_len, len(tokens) - i), 1, -1):
            phr = " ".join(low[i : i + n])
            found, cats = dicts.find(phr, "external")
            if found:
               best = (n, cats)
               break
         if best is None:
            continue
         n, cats = best
         phrase_B[i] = (n, cats)
         for j in range(i + 1, i + n):
            phrase_I[j] = (n, cats)

   for i,tk in enumerate(tokens) :
      tokenFeatures = []
      t = tk.text

      # --- current token features ---
      add_token_features(tokenFeatures, t, t, 0, dicts, feature_flags)

      # --- multi-token dictionary phrase features ---
      if feature_flags.external_phrase and phrase_B[i] is not None:
         n, cats = phrase_B[i]
         tokenFeatures.append(f"externalPhrasePos=B")
         tokenFeatures.append(f"externalPhraseLen={n}")
         for c in cats:
            tokenFeatures.append(f"externalPhrase={c}")
      elif feature_flags.external_phrase and phrase_I[i] is not None:
         n, cats = phrase_I[i]
         tokenFeatures.append(f"externalPhrasePos=I")
         tokenFeatures.append(f"externalPhraseLen={n}")
         for c in cats:
            tokenFeatures.append(f"externalPhrase={c}")

      # --- context window (n-x ... n+x, excluding current token) ---
      for offset in range(-context_window, context_window + 1):
         if offset == 0:
            continue
         j = i + offset
         if j < 0 or j >= len(tokens):
            tokenFeatures.append(boundary_marker(offset))
            continue
         add_token_features(tokenFeatures, tokens[j].text, t, offset, dicts, feature_flags)

      # bigram: current + next (joint context feature)
      if i < len(tokens)-1 and context_window >= 1:
         tokenFeatures.append("bigram="+t.lower()+"_"+tokens[i+1].text.lower())
         # other combinations
         tokenFeatures.append("shapePair="+word_shape(t)+"_"+word_shape(tokens[i+1].text))
         if feature_flags.length_next:
            tokenFeatures.append("lengthNext=%s" % len(tokens[i+1].text))
      elif context_window >= 1:
         # consistent with existing boundary markers for offset=+1
         tokenFeatures.append("EoS")

      # current and previous words together
      if i>0 and context_window >= 1:
         tokenFeatures.append("bigramPrev="+tokens[i-1].text.lower()+"_"+t.lower())
      # previous, current, and next words together
      if i>0 and i<len(tokens)-1 and context_window >= 1:
         tokenFeatures.append("trigramPrevCurrNext="+tokens[i-1].text.lower()+"_"+t.lower()+"_"+tokens[i+1].text.lower())

      sentenceFeatures[i] = tokenFeatures
    
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(
    datafile, outfile, context_window: int = 1, feature_flags: Optional[FeatureFlags] = None
):
   if feature_flags is None:
      feature_flags = FeatureFlags()

   if context_window < 0:
      raise ValueError("context_window must be >= 0")

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
      features = extract_sentence_features(tokens, dicts, context_window, feature_flags)

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
## -- Usage: baseline-NER.py target-file outfile [context-window]
## --
## -- Extracts Drug NE features and writes feature vectors to outfile
## --

if __name__ == "__main__" :
   # file to process
   datafile = sys.argv[1]
   # file where to store results
   featfile = sys.argv[2]
   # optional context window size x (features from n-x to n+x)
   context_window = int(sys.argv[3]) if len(sys.argv) > 3 else 1
   # optional: comma-separated enabled flags; omit for all on; "" for all off
   ff_arg = sys.argv[4] if len(sys.argv) > 4 else None
   feature_flags = (
      parse_feature_flags_param(ff_arg) if len(sys.argv) > 4 else FeatureFlags()
   )

   extract_features(datafile, featfile, context_window, feature_flags)

