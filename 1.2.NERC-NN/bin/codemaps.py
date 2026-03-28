import os
import string
import re
import torch

from dataset import *

# folder where this file is located
THISDIR=os.path.abspath(os.path.dirname(__file__))
# go two folders up and locate "resources" folder there
NERDIR=os.path.dirname(THISDIR)
SOLDIR=os.path.dirname(NERDIR)
MAINDIR=os.path.dirname(SOLDIR)
RESOURCESDIR=os.path.join(MAINDIR, "resources")


# ---------------------------------------------------------------------------
# Feature group registry
# Each entry maps a group name → number of bits it contributes to the vector.
# To add a new group: (1) add an entry here, (2) add a _feat_<name> method.
# ---------------------------------------------------------------------------
FEATURE_SIZES = {
    'ortho':      6,  # is_upper, is_title, is_digit, has_hyphen, has_number, has_punct
    'ext_full':   5,  # DrugBank/HSDB full-name: drug, group, brand, drug_n, any
    'ext_part':   5,  # DrugBank/HSDB partial:   drug, group, brand, drug_n, any
    'drug_morph': 2,  # pharmaceutical suffix / prefix pattern
    'greek':      1,  # contains a greek letter (α, β, γ …)
    'length':     2,  # long word (>9 chars), very short word (≤2 chars)
    'spacy':      4,  # is NOUN/PROPN, is_stop, is_alpha, like_num
}

# Groups that reproduce the original 16-bit vector (used when 'features' is
# absent from params, so existing experiments are not affected).
DEFAULT_FEATURES = ['ortho', 'ext_full', 'ext_part']

# Convenience alias for "use everything"
ALL_FEATURES = list(FEATURE_SIZES.keys())

# Pharmaceutical morphology patterns
_DRUG_SUFFIXES = (
    'ine', 'ide', 'ase', 'ol', 'mab', 'nib', 'vir',
    'pril', 'sartan', 'statin', 'olol', 'cillin', 'mycin', 'tide',
    'afil', 'azole', 'oxacin', 'cycline',
)
_DRUG_PREFIXES = (
    'anti', 'di', 'tri', 'mono', 'poly', 'hydro',
    'chlor', 'methyl', 'deoxy', 'amino', 'nitro',
)
_GREEK_LETTERS = set('αβγδεζηθικλμνξοπρστυφχψω')


class Codemaps:
    # --- constructor: create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, params):
        maxlen = params.get('max_len')
        suflen = params.get('suf_len')

        #---------------------- load external lexicons
        self.external = {}
        self.externalpart = {}
        with open(os.path.join(RESOURCESDIR, "HSDB.txt"), encoding='utf-8') as h:
            for x in h.readlines():
                x = x.strip().lower()
                self.external[x] = {"any"}
                wds = x.split()
                if len(wds) > 1:
                    for w in wds:
                        self.externalpart[w] = {"any"}

        with open(os.path.join(RESOURCESDIR, "DrugBank.txt"), encoding='utf-8') as h:
            for x in h.readlines():
                (n, t) = x.strip().lower().split("|")
                if n in self.external:
                    self.external[n].add(t)
                else:
                    self.external[n] = {t}
                wds = n.split()
                if len(wds) > 1:
                    for w in wds:
                        if w in self.externalpart:
                            self.externalpart[w].add(t)
                        else:
                            self.externalpart[w] = {t}

        #----------------------
        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            feature_set = params.get('features', DEFAULT_FEATURES)
            self.__create_indexs(data, maxlen, suflen, feature_set)

        elif type(data) == str:
            print('Codemaps: ', end='')
            if maxlen is not None or suflen is not None:
                print('Ignoring given params and ', end='')
            print(f'loading index from {data}.idx')
            self.__load(data)

        else:
            print(f'codemaps: Missing max_len and/or suf_len parameters. params={params}')
            exit()

    # --------- Create indexes from training data
    def __create_indexs(self, data, maxlen, suflen, feature_set):
        self.maxlen = int(maxlen)
        self.suflen = int(suflen)

        # Validate feature groups
        unknown = [g for g in feature_set if g not in FEATURE_SIZES]
        if unknown:
            print(f'codemaps: unknown feature group(s): {unknown}. '
                  f'Valid groups: {list(FEATURE_SIZES.keys())}')
            exit()
        self.feature_set = list(feature_set)

        words = set()
        lc_words = set()
        sufs = set()
        labels = set()

        for _, tokens, lab in data.sentences():
            for i, t in enumerate(tokens):
                if t.text.startswith(" "):
                    continue
                words.add(t.text)
                lc_words.add(t.text.lower())
                sufs.add(t.text.lower()[-self.suflen:])
                labels.add(lab[i])

        self.word_index = {w: i+2 for i, w in enumerate(list(words))}
        self.word_index['PAD'] = 0
        self.word_index['UNK'] = 1

        self.lc_word_index = {w: i+2 for i, w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0
        self.lc_word_index['UNK'] = 1

        self.suf_index = {s: i+2 for i, s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0
        self.suf_index['UNK'] = 1

        self.label_index = {t: i+1 for i, t in enumerate(list(labels))}
        self.label_index['PAD'] = 0

    # --------- Load indexes from file
    def __load(self, name):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.feature_set = list(DEFAULT_FEATURES)  # safe default if absent in file

        with open(name + ".idx") as f:
            for line in f.readlines():
                parts = line.split()
                t = parts[0]
                if t == 'MAXLEN':
                    self.maxlen = int(parts[1])
                elif t == 'SUFLEN':
                    self.suflen = int(parts[1])
                elif t == 'FEATURESET':
                    self.feature_set = parts[1:]
                elif t == 'WORD':
                    self.word_index[parts[1]] = int(parts[2])
                elif t == 'LCWORD':
                    self.lc_word_index[parts[1]] = int(parts[2])
                elif t == 'SUF':
                    self.suf_index[parts[1]] = int(parts[2])
                elif t == 'LABEL':
                    self.label_index[parts[1]] = int(parts[2])

    # ---------- Save model and indexes
    def save(self, name):
        with open(name + ".idx", "w") as f:
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            print('FEATURESET', *self.feature_set, file=f)
            for key in self.label_index:
                print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index:
                print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index:
                print('LCWORD', key, self.lc_word_index[key], file=f)
            for key in self.suf_index:
                print('SUF', key, self.suf_index[key], file=f)

    # -------------------------------------------------------------------------
    # Padding / encoding helpers
    # -------------------------------------------------------------------------

    def cut_and_pad(self, tensor_list, pad):
        if len(tensor_list[0].shape) == 1:
            shape = (len(tensor_list), self.maxlen)
        elif len(tensor_list[0].shape) == 2:
            shape = (len(tensor_list), self.maxlen, tensor_list[0].shape[1])
        tensor_list = [s[0:self.maxlen] for s in tensor_list]
        padded = torch.Tensor([]).new_full(shape, pad, dtype=torch.int64)
        for i, s in enumerate(tensor_list):
            for j, f in enumerate(s):
                padded[i, j] = f
        return padded

    def encode_words(self, data):
        enc = [torch.Tensor([self.word_index.get(w.text, self.word_index['UNK'])
                             for w in s])
               for _, s, _ in data.sentences()]
        Xw = self.cut_and_pad(enc, self.word_index['PAD'])

        enc = [torch.Tensor([self.lc_word_index.get(w.text.lower(), self.lc_word_index['UNK'])
                             for w in s])
               for _, s, _ in data.sentences()]
        Xlw = self.cut_and_pad(enc, self.lc_word_index['PAD'])

        enc = [torch.Tensor([self.suf_index.get(w.text.lower()[-self.suflen:],
                                                 self.suf_index['UNK'])
                             for w in s])
               for _, s, _ in data.sentences()]
        Xs = self.cut_and_pad(enc, self.suf_index['PAD'])

        enc = [torch.Tensor([self.features(w) for w in s])
               for _, s, _ in data.sentences()]
        Xf = self.cut_and_pad(enc, 0)

        return [Xlw, Xw, Xs, Xf]

    def encode_labels(self, data):
        enc = [torch.Tensor([self.label_index[lab] for lab in l])
               for _, _, l in data.sentences()]
        return self.cut_and_pad(enc, self.label_index['PAD'])

    # -------------------------------------------------------------------------
    # Size queries
    # -------------------------------------------------------------------------

    def get_n_words(self):      return len(self.word_index)
    def get_n_lc_words(self):   return len(self.lc_word_index)
    def get_n_sufs(self):       return len(self.suf_index)
    def get_n_labels(self):     return len(self.label_index)
    def get_n_features(self):   return sum(FEATURE_SIZES[g] for g in self.feature_set)

    def word2idx(self, w):      return self.word_index[w]
    def lcword2idx(self, w):    return self.lc_word_index[w]
    def suff2idx(self, s):      return self.suf_index[s]
    def label2idx(self, l):     return self.label_index[l]

    def idx2label(self, i):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError

    # -------------------------------------------------------------------------
    # Feature groups — one method per group, named _feat_<group_name>
    # Each returns a plain Python list of ints (0/1).
    # -------------------------------------------------------------------------

    def _feat_ortho(self, form):
        """6 bits: surface orthographic properties."""
        return [
            int(form.isupper()),
            int(form.istitle()),
            int(form.isdigit()),
            int('-' in form),
            int(bool(re.search('[0-9]', form))),
            int(any(c in string.punctuation for c in form)),
        ]

    def _feat_ext_full(self, lcform):
        """5 bits: full-token match against DrugBank / HSDB."""
        ext = self.external.get(lcform, set())
        return [
            int('drug'   in ext),
            int('group'  in ext),
            int('brand'  in ext),
            int('drug_n' in ext),
            int('any'    in ext),
        ]

    def _feat_ext_part(self, lcform):
        """5 bits: token is part of a multi-word entry in DrugBank / HSDB."""
        ext = self.externalpart.get(lcform, set())
        return [
            int('drug'   in ext),
            int('group'  in ext),
            int('brand'  in ext),
            int('drug_n' in ext),
            int('any'    in ext),
        ]

    def _feat_drug_morph(self, lcform):
        """2 bits: common pharmaceutical suffix / prefix patterns."""
        return [
            int(any(lcform.endswith(s) for s in _DRUG_SUFFIXES)),
            int(any(lcform.startswith(p) for p in _DRUG_PREFIXES)),
        ]

    def _feat_greek(self, lcform):
        """1 bit: contains a greek letter (α-blockers, β-lactam, …)."""
        return [int(any(c in _GREEK_LETTERS for c in lcform))]

    def _feat_length(self, form):
        """2 bits: long word (>9 chars), very short word (≤2 chars)."""
        return [int(len(form) > 9), int(len(form) <= 2)]

    def _feat_spacy(self, w):
        """4 bits: spaCy-derived linguistic properties (already computed, free)."""
        return [
            int(w.pos_ in ('NOUN', 'PROPN')),
            int(w.is_stop),
            int(w.is_alpha),
            int(w.like_num),
        ]

    # -------------------------------------------------------------------------
    # Main feature dispatcher — assembles the active feature groups in order
    # -------------------------------------------------------------------------

    def features(self, w):
        """Return the feature vector for token w (a spaCy Token).
        The vector length equals get_n_features() and depends on self.feature_set.
        Returns a zero vector when w is None (used by get_n_features via network init).
        """
        if w is None:
            return [0] * self.get_n_features()

        form   = w.text
        lcform = form.lower()

        bits = []
        for group in self.feature_set:
            if   group == 'ortho':      bits += self._feat_ortho(form)
            elif group == 'ext_full':   bits += self._feat_ext_full(lcform)
            elif group == 'ext_part':   bits += self._feat_ext_part(lcform)
            elif group == 'drug_morph': bits += self._feat_drug_morph(lcform)
            elif group == 'greek':      bits += self._feat_greek(lcform)
            elif group == 'length':     bits += self._feat_length(form)
            elif group == 'spacy':      bits += self._feat_spacy(w)
        return bits
