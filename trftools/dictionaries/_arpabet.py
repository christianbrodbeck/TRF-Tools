"""
Arpabet
-------

https://en.wikipedia.org/wiki/Arpabet

"""
from collections import OrderedDict
import csv
from itertools import product
import json
from os.path import dirname, exists, join


VOWELS = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY',
          'OW', 'OY', 'UH', 'UW')
CONSONANTS = ('B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
              'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')
SILENCE = ('sp', 'sil', '', ' ')
ARPABET = VOWELS + CONSONANTS + (' ',)
# https://en.wikipedia.org/wiki/ARPABET
# 'AH': can also be 'ʌ'
IPA = {
    'AO': 'ɔ', 'AA': 'ɑ', 'IY': 'i', 'UW': '', 'EH': 'ɛ', 'IH': 'ɪ',
    'UH': 'ʊ', 'AH': 'ə', 'AX': 'ə', 'AE': 'æ',
    'EY': 'eɪ', 'AY': 'aɪ', 'OW': 'oʊ', 'AW': 'aʊ', 'OY': 'ɔɪ', 'ER': 'ɝ',
    'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'g',
    'CH': 'tʃ', 'JH': 'dʒ',
    'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð', 'S': 's', 'Z': 'z',
    'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h',
    'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'L': 'l', 'R': 'ɹ',
    'Y': 'j', 'W': 'w',
    ' ': ' ',
}

#  {vowel_with_stress: vowel}
STRIP_STRESS = {'%s%i' % (v, s): v for v, s in product(VOWELS, (0, 1, 2))}
STRIP_STRESS_MAP = {c: c for c in CONSONANTS}
STRIP_STRESS_MAP.update(STRIP_STRESS)

# map phones with or without stress to phones
NORMALIZE_NO_STRESS = {v: v for v in VOWELS}
NORMALIZE_NO_STRESS.update(STRIP_STRESS_MAP)
NORMALIZE_NO_STRESS.update({s: ' ' for s in SILENCE})

# Only silence
SILENCE_MAP = {k: 'silence' if v == ' ' else '' for k, v in
               NORMALIZE_NO_STRESS.items()}

# level 3 (sonorant-syllabic-continuant
SON_nSYL_CON = ('R', 'W', 'Y')
SON_nSYL_nCON = ('L', 'M', 'N', 'NG')
nSON_CON_STR = ('Z', 'ZH', 'S', 'SH')
nSON_CON_nSTR = ('V', 'DH', 'F', 'TH', 'HH')
nSON_nCON_STR = ('JH', 'CH')
nSON_nCON_nSTR = ('B', 'D', 'G', 'P', 'T', 'K')
# level 2
nSON_CON = nSON_CON_STR + nSON_CON_nSTR
nSON_nCON = nSON_nCON_STR + nSON_nCON_nSTR
SYLLABIC = VOWELS
SON_nSYL = SON_nSYL_CON + SON_nSYL_nCON
# level 1
SONORANT = SYLLABIC + SON_nSYL
nSONORANT = nSON_CON + nSON_nCON


########################################################################
# Classification schemes
########################

def expand_mapping(map_in):
    "{1: (2, 3)} -> {2: 1, 3: 1}"
    return {k: v for v, ks in map_in.items() for k in ks}


def generalize_mapping(inner, outer, default=''):
    "generalize_mapping({2: 3}, {1: 2}) -> {1: 3}"
    return {ko: inner.get(ki, default) for ko, ki in outer.items()}


# mapping from aligner phones to different classification schemes
def class_var(mapping, default='', base=NORMALIZE_NO_STRESS):
    gen = expand_mapping(mapping)
    if base is not None:
        gen = generalize_mapping(gen, base, default)
    return gen


PHONE_CLASSIFICATION = {
    'any': {k: '+' if v != ' ' else ' ' for k, v in
            NORMALIZE_NO_STRESS.items()},
    'phone': NORMALIZE_NO_STRESS,
    'silence': SILENCE_MAP,
    'level1': class_var({'son+': SONORANT, 'son-': nSONORANT}),
    'level2': class_var({'son+syl+': VOWELS,
                         'son+syl-': SON_nSYL,
                         'son-con+': nSON_CON,
                         'son-con-': nSON_nCON}),
    'level3': class_var({'son+syl+': VOWELS,
                         'son+syl-con-': SON_nSYL_nCON,
                         'son+syl-con+': SON_nSYL_CON,
                         'son-con+str+': nSON_CON_STR,
                         'son-con+str-': nSON_CON_nSTR,
                         'son-con-str+': nSON_nCON_STR,
                         'son-con-str-': nSON_nCON_nSTR}),
    'mesgarani': class_var({
        'plosive': ('D', 'B', 'G', 'P', 'K', 'T'),
        'fricative': ('SH', 'S', 'Z', 'ZH', 'F', 'TH'),
        'low back': ('AH', 'AY', 'ER', 'L', 'OW', 'OY', 'UW', 'W'),
        'low front': ('AE', 'AW', 'EH', 'EY'),
        'high front': ('DH', 'IH', 'IY', 'UH', 'V', 'Y'),
        'nasal': ('N', 'NG', 'M')}),
}

PERMUTATION_TARGETS = {
    'level1>2': {
        'son+': ('son+syl+', 'son+syl-'),
        'son-': ('son-con+', 'son-con-'),
    },
    'level2>3': {
        'son+syl+': ('son+syl+',),
        'son+syl-': ('son+syl-con-', 'son+syl-con+'),
        'son-con+': ('son-con+str+', 'son-con+str-'),
        'son-con-': ('son-con-str+', 'son-con-str-'),
    },
    'level3>phone': {
        'son+syl+': VOWELS,
        'son+syl-con-': SON_nSYL_nCON,
        'son+syl-con+': SON_nSYL_CON,
        'son-con+str+': nSON_CON_STR,
        'son-con+str-': nSON_CON_nSTR,
        'son-con-str+': nSON_nCON_STR,
        'son-con-str-': nSON_nCON_nSTR},
}


########################################################################
# Feature maps
##############
# - some features can be more than binary (1, 0, -1)
# - diphthongs are labeled according to the first vowel (Phoebe)
FEATURE_VALUES = {'': 0, '0': 0, '1': 1, '-1': -1}


def _path(name, ext):
    path = join(dirname(__file__), 'data', 'arpabet_' + name + ext)
    if not exists(path):
        if exists(name):
            return name
        else:
            raise IOError("No feature table named %s" % (name,))
    return path


def read_feature_table(name):
    path = _path(name, '.tsv')
    out = OrderedDict()
    with open(path, 'rb') as fid:
        reader = csv.reader(fid, 'excel-tab')
        phones = next(reader)[1:]
        assert phones.pop(-1) == 'SIL'
        phones.append(' ')
        for line in reader:
            assert len(line) == len(phones) + 1, "missing entries"
            out[line[0]] = {p: FEATURE_VALUES[i] for p, i in zip(phones, line[1:])}
    return out


def read_feature_groups(name):
    path = _path(name, '.json')
    with open(path, 'r') as fid:
        return json.load(fid)


def print_feature_table(name):
    table = read_feature_table(name)
    for c in ARPABET:
        desc = "%2s (%s)" % (c, IPA[c])
        features = ', '.join(k for k, v in table.items() if v[c])
        print("%8s: %s" % (desc, features))
