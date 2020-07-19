"""Convert unicode text to label for force aligning."""
from collections import defaultdict
import fnmatch
from glob import glob
from itertools import chain, zip_longest, repeat
import json
from math import ceil
import os
from pathlib import Path
import string
from typing import Union

from eelbrain import fmtxt, Dataset
import numpy as np
import textgrid

from .._utils import LookaheadIter
from ..dictionaries import read_cmupd, read_dict, combine_dicts
from ..dictionaries._arpabet import SILENCE
from ..dictionaries._dict import APOSTROPHE_I
from ._text import text_to_words


PUNC = [s.encode('ascii') for s in string.punctuation + "\n\r"]


class Realization:
    """Pronunciation and corresponding graph sequence for a word in a TextGrid

    Notes
    -----
    For silence, ``realization.graphs == ' '``.
    """
    __slots__ = ('graphs', 'phones', 'pronunciation', 'times', 'tstop')

    def __init__(self, phones, times, graphs, tstop):
        self.phones = phones  # tuple of str
        self.pronunciation = ' '.join(phones)
        self.times = times  # tuple of float, phone onset times
        self.graphs = graphs
        self.tstop = tstop

    def __repr__(self):
        args = (self.phones, self.times, self.graphs, self.tstop)
        args = ', '.join(map(repr, args))
        return f"Realization({args})"

    def strip_stress(self):
        "Strip stress information (numbers 0/1/2 on vowels)"
        phones = tuple([p.rstrip('012') for p in self.phones])
        return Realization(phones, self.times, self.graphs, self.tstop)


class TextGrid:
    "TextGrid representation for regressors"
    def __init__(self, grid_file, tmin=0., tstep=0.001, n_samples=None, word_tier='words', phone_tier='phones'):
        realizations = textgrid_as_realizations(grid_file, word_tier, phone_tier)
        self._n_samples_arg = n_samples

        if n_samples is None:
            n_samples = int(ceil((realizations[-1].tstop - tmin) / tstep))
        else:
            t_stop = n_samples * tstep + tmin
            r_last = realizations[-1]
            if r_last.tstop < t_stop:
                realizations.append(Realization((' ',), (r_last.tstop,), ' ', t_stop))
            elif r_last.times[-1] > t_stop:
                print("Warning: dropping %.3f s from %s" % (r_last.times[-1] - t_stop, grid_file))

        self.tmin = tmin
        self.tstep = tstep
        self.realizations = realizations
        self.n_samples = n_samples
        self._grid_file = grid_file
        self._stop = int(round((realizations[-1].tstop - tmin) / tstep))

    def __repr__(self):
        args = [repr(self._grid_file)]
        if self.tmin:
            args.append("tmin=%r" % self.tmin)
        if self.tstep != 0.001:
            args.append("tstep=%r" % self.tstep)
        if self._n_samples_arg:
            args.append("n_samples=%r" % self._n_samples_arg)
        return "TextGrid(%s)" % ', '.join(args)

    def split_by_apostrophe(self):
        """SUBTLEX respresents "ISN'T" as "INS" + "T"

        This method replace the TextGrid's realization that contain an
        apostrophe accordingly
        """
        new = []
        for realization in self.realizations:
            if "'" in realization.graphs:
                g1, g2 = realization.graphs.split("'")
                try:
                    i_split = APOSTROPHE_I[g2]
                except KeyError:
                    raise KeyError(f"{g2!r} ({realization}")
                ps = realization.phones[:i_split]
                ts = realization.times[:i_split]
                tstop = realization.times[i_split]
                new.append(Realization(ps, ts, g1, tstop))
                ps = realization.phones[i_split:]
                ts = realization.times[i_split:]
                tstop = realization.tstop
                new.append(Realization(ps, ts, g1, tstop))
            else:
                new.append(realization)
        self.realizations = new

    def align(self, words, values, silence=0, unknown=None):
        """Align values to the words in the textgrid"""
        n_words = len(words)
        assert len(values) == n_words
        grid_words = [r.graphs for r in self.realizations]
        i_next = last_match_i = last_match_i_grid = 0  # i_next: start of unused words in ``words``
        out = []
        for i_grid, grid_word in enumerate(grid_words):
            if grid_word == ' ':
                out.append(silence)
            else:
                for i in range(i_next, n_words):
                    word_i = words[i]
                    if word_i == unknown:
                        break
                    elif grid_word == 'CANNOT' and word_i == 'can' and words[i + 1] == 'not':
                        assert words[i + 2] != 'not' and words[i + 3] != 'not'
                        break
                    word_i = word_i.strip("'")
                    if word_i.upper() == grid_word:
                        break
                else:
                    n = min(9, len(grid_words) - i_grid, len(words) - last_match_i)
                    ds = Dataset()
                    ds['grid_words'] = grid_words[last_match_i_grid: last_match_i_grid + n]
                    ds['words'] = words[last_match_i: last_match_i + n]
                    raise ValueError(f"Can't align words to {self._grid_file} after word {i_next}:\n{ds}")
                out.append(values[i])
                last_match_i = i
                last_match_i_grid = i_grid
                i_next = i + 1
        return out

    def get_indexes(self, feature='phone', stop=True):
        """Event boundary indexes (in sample)

        Parameters
        ----------
        feature : str
            Feature for which to retrieve index:
             - 'phone': phonemes (including silence)
             - 'word': words (including silence)
             - 'word-up': onset of the morphological uniqueness point phoneme
               (silence onset for silence).
        stop : bool
            Include stop index (for intervals).
        """
        tmin = self.tmin
        tstep = self.tstep
        if feature == 'phone':
            times = (t for w in self.realizations for t in w.times)
        elif feature == 'word':
            times = (w.times[0] for w in self.realizations)
        else:
            raise ValueError("feature=%r" % (feature,))
        indexes = [int(round((t - tmin) / tstep)) for t in times]
        if stop:
            indexes.append(self._stop)

        return indexes

    def categories_to_array(self, tokens, codes, location, weights=None, index='phone'):
        """Convert NUTS of categories to binary category x time array

        Parameters
        ----------
        tokens : sequence of str
            Category assignment for each realization (i.e., includes silence).
        codes : {str: int}
            Mapping tokens in ``tokens`` to row indexes.
        location : 'fill' | 'onset' | 'offset'
            Wehere to place the value relative to the interval.
        weights : sequency of float (optional)
            Assign a weight to each entry of ``tokens`` (default is 1)
        index : str
            Feature for boundary location:
             - 'phone': phonemes
             - 'word': words
             - 'word$relocate': word boundaries with permuted location
             - 'word-up': morphological uniqueness point phoneme (or silence)
        """
        if location == 'fill':
            indexes = self.get_indexes(index)
            indexes = [slice(indexes[i], indexes[i+1]) for i in range(len(indexes) - 1)]
        elif location == 'onset':
            indexes = self.get_indexes(index, stop=False)
        elif location == 'offset':
            indexes = self.get_indexes(index)[1:]
        else:
            raise ValueError("location=%s" % repr(location))

        if weights is None:
            weights = repeat(1, len(indexes))

        out = np.zeros((max(codes.values()) + 1, self.n_samples))
        for value, index, weight in zip(tokens, indexes, weights):
            if value in codes:
                out[codes[value], index] = weight

        return out

    @staticmethod
    def categories_to_codes(values, codes, weights=None):
        return _categories_to_codes(values, codes, weights)

    def print(self, segmentation=None):
        """Print text with pronunciation

        Parameters
        ----------
        segmentation : Lexicon
            Display morphological segmentation form this lexicon.
        """
        lines = ['', '']
        for r in self.realizations:
            graphs = r.graphs
            pronunciation = r.pronunciation
            if segmentation:
                words = segmentation.lookup(graphs.lower())
                if words:
                    word = words[0]
                    graphs = '+'.join(word.segmentation)
            n = max(map(len, (graphs, pronunciation))) + 2
            if len(lines[-1]) + n > 80:
                lines.extend(('', ''))
            lines[-2] += graphs.ljust(n)
            lines[-1] += pronunciation.ljust(n)
        print('\n'.join(lines))

    def values_to_array(self, values, value_location, index='phone'):
        indexes = self.get_indexes(index)
        if values.ndim == 1:
            out = _values_to_array(values, indexes, self.n_samples, value_location)
        elif values.ndim == 2:
            out = np.zeros((len(values), self.n_samples))
            for array, row_values in zip(out, values):
                _values_to_array(row_values, indexes, self.n_samples, value_location, array)
        else:
            raise ValueError("Need array with 1 or 2 dimensions, got shape=%s" %
                             (values.shape,))
        return out


def gentle_to_grid(gentle_file, out_file=None):
    "Convert *.json file from Gentle to Praat TextGrid"
    if '*' in gentle_file:
        if out_file is not None:
            raise TypeError("out can not be set during batch-conversion")
        for filename in glob(gentle_file):
            gentle_to_grid(filename)
        return

    gentle_file = Path(gentle_file)
    if out_file is None:
        out_file = gentle_file.with_suffix('.TextGrid')
    else:
        out_file = Path(out_file)
        if out_file.suffix.lower() != '.textgrid':
            out_file = out_file.with_suffix('.TextGrid')

    with gentle_file.open() as fid:
        g = json.load(fid)
    all_words = g['words']

    # check for words that were not successfully aligned
    words = [w for w in all_words if w['case'] == 'success']
    if len(words) < len(all_words):
        log = fmtxt.Table('rll')
        log.cells('Time', 'Word', 'Issue')
        log.midrule()
        for i, word in enumerate(all_words):
            if word['case'] == 'success':
                if word['alignedWord'] == '<unk>':
                    issue = 'OOV'
                    t_start = word['start']
                else:
                    continue
            else:
                issue = word['case']
                while i:
                    if 'end' in all_words[i-1]:
                        t_start = all_words[i-1]['end']
                        break
                    i -= 1
                else:
                    t_start = 0
            log.cells(f'{t_start:.3f}', word['word'], issue)
        print(log)
        log.save_tsv(out_file.with_suffix('.log'))

    # round times
    for word in words:
        word['start'] = round(word['start'], 3)
        word['end'] = round(word['end'], 3)

    # avoid overlapping words
    last_start = words[-1]['end'] + 1
    for word in reversed(words):
        if word['end'] > last_start:
            word['end'] = last_start
            if word['start'] >= last_start:
                word['start'] = last_start - .001
        last_start = word['start']

    # build textgrid
    phone_tier = textgrid.IntervalTier('phones')
    word_tier = textgrid.IntervalTier('words')
    last_word_i = len(words) - 1
    for i, word in enumerate(words):
        if not word['case'] == 'success':
            continue
        t = word['start']
        word_tstop = word['end']
        # prevent the word from overlapping with the next word
        if i < last_word_i and words[i + 1]['case'] == 'success':
            word_tstop = min(word_tstop, words[i + 1]['start'])
        # only add words with positive duration (Praat can't handle others)
        if t >= word_tstop:
            continue
        # add word and phones
        word_tier.add(t, word_tstop, word['word'])
        for phone in word.get('phones', ()):
            tstop = min(round(t + phone['duration'], 3), word_tstop)
            if t >= tstop:
                continue
            phone_tier.add(t, tstop, phone['phone'].split('_')[0].upper())
            t = tstop
    grid = textgrid.TextGrid()
    grid.extend((phone_tier, word_tier))
    grid.write(out_file)


def _load_tier(grid, tier: str = 'phones'):
    """Load one or more tiers as textgrid Tier object"""
    names = [name for name in grid.getNames() if fnmatch.fnmatch(name.lower(), tier.lower())]
    if len(names) != 1:
        available = ', '.join(grid.getNames())
        raise IOError(f"{len(names)} tiers match {tier!r} in {grid.name or grid}. Availabe tiers: {available}")
    return grid.getFirst(names[0])


def dict_lookup(pronunciations, word):
    try:
        return pronunciations[word]
    except KeyError:
        raise KeyError(f"No pronunciation for {key}")


def grid_to_dict(grid):
    """Generate dict from words in a text-grid"""
    realizations = textgrid_as_realizations(grid)
    words = defaultdict(set)
    for r in realizations:
        if r.graphs in SILENCE:
            continue
        words[r.graphs.upper()].add(' '.join(r.phones))
    return words


def fix_word_tier(
        grid_filename: Path,
        text_filename: Path = None,
        out: Path = None,
        strip_stress: bool = True,
        pronunciation_dict: Union[str, dict] = None,  # add pronunciations
        silence: list = (),  # additional silence marks
):
    """Fix word boundaries after manually aligning phones tier

    Parameters
    ----------
    grid_filename: Path
        TextGrid to fix.
    text_filename: Path
        Transcript, if available.
    out: Path
        Destination for the fixed TextGrid. Default is ``grid_filename`` with
        ``-fixed`` tag added.
    strip_stress: bool
        Remove stress information (default ``True``).
    pronunciation_dict: str | dict
        Add pronunciations (in addition to the CMU pronouncing dictionary,
        which is used by default). Can be a path to a *.dict file, or a
        dictionary of pronounciations (e.g., ``{
        'ADEQUACY': 'AE D AH K W AH S IY',
        'ADEQUATE': ['AE D AH K W AH T', 'AE D AH K W EY T'],
        }``).
    silence : list of str
        Additional silence marks.
    """
    if out is None:
        grid_path = Path(grid_filename)
        out = grid_path.parent / f'{grid_path.stem}-fixed.TextGrid'
    grid = textgrid.TextGrid.fromFile(grid_filename)
    if len(grid.getList('phones')) != 1 or len(grid.getList('words')) != 1:
        raise ValueError(f"File has non-unique tier-names: {grid_filename}")
    # silence marks
    if isinstance(silence, str):
        silence = [silence]
    silence = {s for s in chain(SILENCE, silence) if s}
    # pronunciation dictionaries
    if pronunciation_dict is None:
        pronunciation_dict = {}
    elif isinstance(pronunciation_dict, str):
        pronunciation_dict = read_dict(pronunciation_dict, strip_stress)
    else:
        pronunciation_dict = {k: [v] if isinstance(v, str) else v for k, v in pronunciation_dict.items()}
    cmu_dict = read_cmupd(strip_stress, apostrophe=False)
    pronunciation_dict = combine_dicts([cmu_dict, pronunciation_dict])
    phone_tier = grid.getFirst('phones')
    word_tier = grid.getFirst('words')
    # clean up phones in phone tier
    for phone in phone_tier:
        phone.mark = phone.mark.strip()
        if phone.mark in silence:
            phone.mark = ''
        elif strip_stress:
            phone.mark = phone.mark.rstrip('012')
    # transcript
    if text_filename:
        words = read_transcript(text_filename).split()
    else:
        words = [i.mark for i in word_tier.intervals if i.mark not in SILENCE]
    del word_tier.intervals[:]
    word_iter = LookaheadIter((w, dict_lookup(pronunciation_dict, w)) for w in words)
    phone_iter = LookaheadIter(phone_tier)
    last_max_time = 0.
    word, phone_reprs = next(word_iter)
    phone_buf = ''
    # for debugging:
    last_word = None
    # alternative:  keep parallel representations whenever there is ambiguity?
    for phone in phone_iter:
        if phone.mark == '':
            if phone_buf:
                raise ValueError(f"Unknown sequence ({phone.minTime:.2f}): {phone_buf}")
            word_tier.add(phone.minTime, phone.maxTime, '')
            last_max_time = phone.maxTime
        elif not phone.mark.isalnum():
            raise ValueError(f"Phone {phone.mark!r} in {phone} is not alpha-numeric")
        else:
            phone_buf = phone.mark if not phone_buf else phone_buf + ' ' + phone.mark
            if phone_buf in phone_reprs:
                # Check whether there might be multiple matches, e.g., and -> {AE N, AE N D}
                next_phone = phone_iter.lookahead()
                if next_phone:
                    if phone_buf + ' ' + next_phone.mark in phone_reprs:
                        next_phone_buf = next_phone.mark
                        # check whether we can build the next word
                        next_word = word_iter.lookahead()
                        # if there is no next word the next phoneme must be ours
                        if not next_word:
                            continue
                        next_word, next_phone_reprs = next_word
                        can_build_next_word = False
                        i = 1
                        while any(p.startswith(next_phone_buf) for p in next_phone_reprs):
                            if next_phone_buf in next_phone_reprs:
                                can_build_next_word = True
                                break
                            i += 1
                            next_phone = phone_iter.lookahead(i)
                            if not next_phone or next_phone.mark in SILENCE:
                                break
                            next_phone_buf = next_phone_buf + ' ' + next_phone.mark
                        if not can_build_next_word:
                            continue

                # close the current word
                word_tier.add(last_max_time, phone.maxTime, word)
                last_word = word, phone_buf, phone_reprs
                phone_buf = ''
                last_max_time = phone.maxTime
                try:
                    word, phone_reprs = next(word_iter)
                except StopIteration:
                    break
            elif len(phone_buf) > max(len(p) for p in phone_reprs):
                if last_word:
                    l_word, l_phone_buf, l_phone_reprs = last_word
                    last_desc = f"; previous word was {l_word} pronounced {l_phone_buf} with known representation(s)  {', '.join(l_phone_reprs)}"
                else:
                    last_desc = ''
                raise ValueError(f"Unknown sequence {phone_buf} encountered at {phone.minTime:.2f} s while looking for word {word} with known representation(s) {', '.join(phone_reprs)}{last_desc}")
    if phone_buf:
        # last word incomplete?
        if any(phone_repr.startswith(phone_buf) for phone_repr in phone_reprs):
            word_tier.add(last_max_time, phone.maxTime, word)
        else:
            raise ValueError(f"Unknown sequence (end): {phone_buf}")
    grid.write(out)


def textgrid_as_realizations(grid, word_tier='words', phone_tier='phones'):
    """Load a TextGrid as a list of Realizations"""
    if isinstance(grid, (str, Path)):
        if isinstance(grid, str) and not grid.lower().endswith('.textgrid'):
            grid += '.TextGrid'
        grid = textgrid.TextGrid.fromFile(grid, grid)
    words = _load_tier(grid, word_tier)
    phones = _load_tier(grid, phone_tier)
    out = []
    phones = list(phones)
    for word in words:
        word_phones = ()
        word_times = ()
        while phones and phones[0].minTime < word.maxTime:
            phone = phones.pop(0)
            word_phones += (phone.mark,)
            word_times += (phone.minTime,)

        if word.mark in SILENCE:
            if not all(p in SILENCE for p in word_phones):
                raise ValueError("Silence word tag (%r) but non-silent phones "
                                 "%r" % (word.mark, word_phones))
            out.append(Realization((' ',), (word.minTime,), ' ', word.maxTime))
        else:
            out.append(Realization(word_phones, word_times, word.mark,
                                   word.maxTime))
    return out


def read_transcript(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.txt':
        return text_to_words(open(filename, 'r').read())
    elif ext == '.lab':
        return open(filename, 'r').read()
    else:
        raise ValueError(f"Unknown extension for transcript file: {ext}")


def _categories_to_codes(values, codes, weights=None):
    """Convert NUTS of categories to binary category x time array

    Parameters
    ----------
    values : sequence of str
        Values.
    codes : {str: int}
        Mapping values in ``values`` to row indexes.
    weights : sequency of float (optional)
        Assign a weight to each entry of ``values`` (default is 1)
    """
    try:
        length = len(values)
    except TypeError:
        values = tuple(values)
        length = len(values)

    if weights is None:
        weights = repeat(1, length)

    out = np.zeros((max(codes.values()) + 1, length))
    for i, value, weight in zip_longest(range(length), values, weights):
        if value in codes:
            out[codes[value], i] = weight

    return out


def _values_to_array(values, indexes, length, location, out=None):
    """Convert NUTS to UTS

    Parameters
    ----------
    values : sequence of scalar
        Values.
    indexes : sequence of int, length of values + 1
        Intervals at which values apply
    length : int
        Length of the array.
    location : 'fill' | 'onset' | 'offset'
        Wehere to place the value relative to the interval.
    """
    assert len(indexes) == len(values) + 1
    if out is None:
        out = np.zeros((1, length))
    else:
        out.shape = (1, length)

    if location == 'fill':
        indexes_ = (slice(start, stop) for start, stop in
                    zip(indexes, indexes[1:]) if start < length)
    elif location == 'onset':
        indexes_ = (i for i in indexes if i < length)
    elif location == 'offset':
        indexes_ = (i - 1 for i in indexes[1:] if i <= length)
    else:
        raise ValueError("location=%s" % repr(location))

    for value, index in zip(values, indexes_):
        out[0, index] = value
    return out


def phoneme_mask(index, realizations):
    """Find phoneme mask for ``index`` (always exclude silence)

    Parameters
    ----------
    index : None | int | slice
        The index to apply.

        - ``None`` to include all (non-silence) phonemes
        - ``int`` to include the ``index``-th phoneme in each word (counting
          starts with ``0``)
        - ``slice`` to include multiple

    realizations : list of Realization
        Text to process.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask array with one entry for each phoneme.
    """
    silence = (-1,)
    count = np.hstack([silence if r.graphs == ' ' else np.arange(len(r.phones)) for r in realizations])
    if isinstance(index, int):
        return count == index
    elif index is None:
        return count >= 0
    elif not isinstance(index, slice):
        raise TypeError(f"index={index}")
    elif index.step is not None and index.step != 1:
        raise NotImplementedError(f"index={index} (step other than 1)")
    mask = count >= (index.start or 0)
    if index.stop is not None:
        mask &= count < index.stop
    return mask
