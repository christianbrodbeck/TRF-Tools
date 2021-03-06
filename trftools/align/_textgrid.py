"""Convert unicode text to label for force aligning."""
from collections import defaultdict
from dataclasses import dataclass, field, replace
import fnmatch
from glob import glob
from itertools import chain, product, repeat, zip_longest
import json
from math import ceil
import os
from pathlib import Path
import string
from typing import Any, Iterator, List, Sequence, Union, Tuple

from eelbrain import fmtxt, Dataset, Factor, Var
from eelbrain._data_obj import FactorArg, asfactor
from eelbrain._types import PathArg
import numpy as np
import textgrid

from .._utils import LookaheadIter
from ..dictionaries import read_cmupd, read_dict, combine_dicts
from ..dictionaries._arpabet import SILENCE
from ._text import text_to_words


PUNC = [s.encode('ascii') for s in string.punctuation + "\n\r"]


class TextGridError(Exception):

    def __init__(self, issue):
        self.issue = issue


@dataclass
class Realization:
    """Pronunciation and corresponding graph sequence for a word in a TextGrid

    Notes
    -----
    Silence is represented as ``' '`` (single space) for ``realization.graphs``
    and ``realization.pronunciation``.
    """
    # __slots__ = ('graphs', 'phones', 'pronunciation', 'times', 'tstop')
    phones: Tuple[str, ...]  # arpabet phones
    times: Tuple[float, ...]  # phone onset times
    graphs: str
    tstop: float
    pronunciation: str = field(init=False)

    def __post_init__(self):
        if len(self.phones) == 0:
            raise ValueError("Word without phones")
        self.pronunciation = ' '.join(self.phones)

    def strip_stress(self):
        "Strip stress information (numbers 0/1/2 on vowels)"
        return replace(self, phones=tuple([p.rstrip('012') for p in self.phones]))


class TextGrid:
    "TextGrid representation for regressors"
    def __init__(
            self,
            realizations: List[Realization],
            tmin: float = 0.,
            tstep: float = 0.001,
            n_samples: int = None,
            name: str = None,
    ):
        self._n_samples_arg = n_samples
        if n_samples is None:
            n_samples = int(ceil((realizations[-1].tstop - tmin) / tstep))
        else:
            t_stop = n_samples * tstep + tmin
            r_last = realizations[-1]
            if r_last.tstop < t_stop:
                realizations.append(Realization((' ',), (r_last.tstop,), ' ', t_stop))

        self.tmin = tmin
        self.tstep = tstep
        self.realizations = realizations
        self.n_samples = n_samples
        self._name = name
        self._stop = int(round((realizations[-1].tstop - tmin) / tstep))
        self.tstop = self.realizations[-1].tstop

    @classmethod
    def from_file(
            cls,
            grid_file: PathArg,
            tmin: float = 0.,
            tstep: float = 0.001,
            n_samples: int = None,
            word_tier: str = 'words',
            phone_tier: str = 'phones',
    ):
        """Load ``*.TextGrid`` file

        Silence tags are normalized to ``' '`` (single space) for phones and
        words. Tags interpreted as silence: ``'sp', 'sil', 'brth', '', ' '``.

        Errors are raised when:

        - a silence word contains non-silence phones
        - a non-silence word contains silence phones
        """
        grid_file = Path(grid_file)
        realizations = textgrid_as_realizations(grid_file, word_tier, phone_tier)
        return cls(realizations, tmin, tstep, n_samples, grid_file.name)

    def __repr__(self):
        args = [repr(self._name)]
        if self.tmin:
            args.append("tmin=%r" % self.tmin)
        if self.tstep != 0.001:
            args.append("tstep=%r" % self.tstep)
        if self._n_samples_arg:
            args.append("n_samples=%r" % self._n_samples_arg)
        return "<TextGrid %s>" % ', '.join(args)

    @property
    def times(self) -> Iterator[float]:
        "All time-stamps in the TextGrid"
        return chain.from_iterable(r.times for r in self.realizations)

    @property
    def phones(self) -> Iterator[str]:
        "All phones in the TextGrid"
        return chain.from_iterable(r.phones for r in self.realizations)

    def table(self, t_start: float = None, t_stop: float = None):
        "fmtxt.Table representation"
        if t_start is None:
            t_start = self.tmin
        if t_stop is None:
            t_stop = self.realizations[-1].tstop
        table = fmtxt.Table('rrll')
        table.cells('#', 'Time', 'Word', 'Phone')
        for i, r in enumerate(self.realizations):
            if r.phones:
                word = r.graphs
                for time, phone in zip(r.times, r.phones):
                    if t_start <= time <= t_stop:
                        table.cells(i, f'{time:.3f}', word, phone)
                    i = word = ''
            # elif t_start <= r.tstop <= t_stop:
            #     table.cells(i, f'{time:.3f}', word, phone)
        return table

    def split_by_apostrophe(self):
        """Split words with apostrophe

        Language models often represent words containing apostrophe as two
        words, for example:

         - he's ->  he 's
         - isn't -> is n't

        This method returns a new TextGrid in which realizations with
        apostrophe are split accordingly.
        """
        # e.g. YOU'D -> Y UW / D
        valid = {'D', 'M', 'S', 'T', 'LL', 'RE', 'VE'}
        new = []
        for realization in self.realizations:
            if "'" in realization.graphs:
                if realization.graphs.upper().endswith("N'T"):
                    ig = -3
                    if realization.pronunciation.endswith('AH N T'):
                        ip = -3
                    else:
                        ip = -2
                else:
                    ig = realization.graphs.index("'")
                    if ig == 0:
                        new.append(replace(realization, graphs=realization.graphs[1:]))
                        continue
                    elif ig == len(realization.graphs) - 1:
                        new.append(replace(realization, graphs=realization.graphs[:-1]))
                        continue
                    elif realization.graphs[ig+1:].upper() in valid:
                        ip = -1
                    else:
                        raise ValueError(f"Unknown word: {realization.graphs!r} ({realization})")
                ps = realization.phones[:ip]
                ts = realization.times[:ip]
                graphs = realization.graphs[:ig]
                tstop = realization.times[ip]
                new.append(Realization(ps, ts, graphs, tstop))
                ps = realization.phones[ip:]
                ts = realization.times[ip:]
                graphs = realization.graphs[ig:]
                tstop = realization.tstop
                new.append(Realization(ps, ts, graphs, tstop))
            else:
                new.append(realization)
        return TextGrid(new, self.tmin, self.tstep, self.n_samples, self._name)

    def strip_stress(self):
        """Remove stress digits from all phones (``K AA1 R`` -> ``K AA R``)"""
        realizations = [r.strip_stress() for r in self.realizations]
        return TextGrid(realizations, self.tmin, self.tstep, self.n_samples, self._name)

    def align_word_dataset(
            self,
            ds: Dataset,
            words: FactorArg = 'word',
    ) -> Dataset:
        """Align ``ds`` to the TextGrid

        Parameters
        ----------
        ds
            Dataset with data to align.
        words
            Words in ``ds`` to use to align to the TextGrid words.

        Returns
        -------
        aligned_ds
            Dataset with the variables in ``ds`` aligned to the TextGrid,
            including time stamps and TextGrid words.
        """
        words_ = asfactor(words, ds=ds)
        index = self._align_index(words_, silence=-1, missing=-2)
        out = Dataset({
            'time': Var([r.times[0] for r in self.realizations]),
            'grid_word': Factor([r.graphs for r in self.realizations]),
        }, info={'tstop': self.realizations[-1].tstop})
        for key, variable in ds.items():
            if isinstance(variable, (Var, Factor)):
                values = dict(enumerate(variable))
                if isinstance(variable, Var):
                    values[-1] = values[-2] = False  # coerced to 0 unless all values are boolean
                    out[key] = Var([values[i] for i in index])
                else:
                    values[-1] = values[-2] = ''
                    out[key] = Factor([values[i] for i in index], random=variable.random)
        return out

    def align(
            self,
            words: Sequence[str],
            values: Sequence[Any],
            silence: Any = 0,
            unknown: str = None,
    ) -> List[Any]:
        """Align ``words`` to the textgrid and sort ``values`` accordingly

        Parameters
        ----------
        words
            The words to align (not case-sensitive). Individual words in
            ``words`` can be skipped, but all words in the TextGrid need to
            occur exactly in ```words``.
        values
            Values corresponding to ``words``.
        silence
            Value to append to the output for silence in the TextGrid.
        unknown
            String to signify unknown words in ``words`` (able to pair with any
            word in the TextGrid).

        Returns
        -------
        aligned_values
            Values from ``values`` aligned with the TextGrid's ``realizations``.
        """
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
                    raise ValueError(f"Can't align words to {self._name} after word {i_next}:\n{ds}")
                out.append(values[i])
                last_match_i = i
                last_match_i_grid = i_grid
                i_next = i + 1
        return out

    def _align_index(
            self,
            words: Sequence[str],
            silence: Any = None,
            missing: Any = None,
            search_distance: int = 6,
    ) -> Sequence:
        """Index into ``words``"""
        # search_order
        j_pairs = list(product(range(search_distance), repeat=2))
        j_pairs.pop(0)
        j_pairs.sort(key=lambda x: x[0]**2 + x[1]**2)
        # input sequences
        words_ = [word.upper() for word in words]
        n_words = len(words_)
        grid_words = [r.graphs.upper() for r in self.realizations]
        n_grid = len(grid_words)
        # counters
        i_grid = i_word = 0  # i_next: start of unused words in ``words``
        out = []
        while i_grid < n_grid:
            # silence
            if grid_words[i_grid] == ' ':
                out.append(silence)
                i_grid += 1
                continue
            # direct match
            if grid_words[i_grid] == words_[i_word]:
                out.append(i_word)
                i_grid += 1
                i_word += 1
                continue
            # grid search for closest match
            for j_grid, j_word in j_pairs:
                if grid_words[i_grid + j_grid] == words_[i_word + j_word]:
                    break
            else:
                # informative error message
                start = min([i_grid, i_word, 2])
                stop = min([10, n_grid - i_grid, n_words - i_word])
                ds = Dataset()
                ds['grid_words'] = grid_words[i_grid - start: i_grid + stop]
                ds['words'] = words[i_word - start: i_word + stop]
                raise ValueError(f"No match within search_distance {search_distance}:\n{ds}")
            # need to fill in one value for each skipped grid
            ii_word = 0
            for ii_grid in range(j_grid):
                if grid_words[i_grid + ii_grid] == ' ':
                    out.append(silence)
                elif ii_word < j_word:
                    out.append(i_word + ii_word)
                    ii_word += 1
                else:
                    out.append(missing)
            # append the next match
            out.append(i_word + j_word)
            i_grid += j_grid + 1
            i_word += j_word + 1
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

    # find valid words
    words = g['words']
    n_issues = 0
    for i, word in enumerate(words):
        if word['case'] == 'success':
            if word['alignedWord'] == '<unk>':
                n_issues += 1
                word['issue'] = 'OOV'
            else:
                word['issue'] = None
        else:
            n_issues += 1
            word['issue'] = word['case']

    # add missing times
    last_end = 0
    not_in_audio_words = []  # buffer
    for word in words:
        if 'start' in word:
            if not_in_audio_words:
                duration = word['start'] - last_end
                for j, word_ in enumerate(not_in_audio_words):
                    word_['start'] = last_end + j * duration
                    word_['end'] = last_end + (j+1) * duration
                not_in_audio_words = []
            last_end = word['end']
        else:
            not_in_audio_words.append(word)
    for word in not_in_audio_words:
        word['start'] = last_end
        word['end'] = last_end = last_end + 0.100

    # round times
    for word in words:
        word['start'] = round(word['start'], 3)
        word['end'] = round(word['end'], 3)

    # avoid overlapping words
    last_start = words[-1]['end'] + 1
    for word in reversed(words):
        if word['end'] > last_start:
            word['end'] = last_start
        if word['start'] >= word['end']:
            word['start'] = word['end'] - .001
        last_start = word['start']
        # gentle seems to work at 10 ms resolution
        if word['end'] - word['start'] < 0.015 and 'issue' not in word:
            word['issue'] = 'short'

    # log issues
    if n_issues:
        log = fmtxt.Table('rrrll')
        log.cell('Time')
        log.cell('Duration', width=2)
        log.cells('Word', 'Issue')
        log.midrule()
        for word in words:
            if word['issue']:
                duration = word['end'] - word['start']
                d_marker = '*' if duration < 0.015 else ''
                log.cells(f"{word['start']:.3f}", d_marker, f"{duration:.3f}", word['word'], word['issue'])
        print(log)
        log.save_tsv(out_file.with_suffix('.log'))

    # build textgrid
    phone_tier = textgrid.IntervalTier('phones')
    word_tier = textgrid.IntervalTier('words')
    for i, word in enumerate(words):
        t = word['start']
        word_tstop = word['end']
        # add word
        word_tier.add(t, word_tstop, word['word'])
        # make sure we have at least one phone
        phones = word.get('phones', ())
        if not phones:
            phones = ({'phone': '', 'duration': word['end'] - word['start']},)
        # add phones
        for phone in phones:
            tstop = min(round(t + phone['duration'], 3), word_tstop)
            if t >= tstop:
                continue
            mark = phone['phone'].split('_')[0].upper()
            if mark == 'OOV':
                continue
            phone_tier.add(t, tstop, mark)
            t = tstop
    grid = textgrid.TextGrid()
    grid.extend((phone_tier, word_tier))
    grid.write(out_file)


def _load_tier(
        grid: textgrid.TextGrid,
        tier: str = 'phones',
        clean: bool = True,
):
    """Load one or more tiers as textgrid Tier object"""
    names = [name for name in grid.getNames() if fnmatch.fnmatch(name.lower(), tier.lower())]
    if len(names) != 1:
        available = ', '.join(grid.getNames())
        raise IOError(f"{len(names)} tiers match {tier!r} in {grid.name or grid}. Availabe tiers: {available}")
    tier = grid.getFirst(names[0])
    if clean:
        for item in tier:
            item.mark = item.mark.strip()
            if item.mark in SILENCE:
                item.mark = ' '
    return tier


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


def textgrid_as_realizations(grid, word_tier='words', phone_tier='phones', strict=True):
    """Load a TextGrid as a list of Realizations"""
    if isinstance(grid, (str, Path)):
        path = grid
        if isinstance(path, str) and not path.lower().endswith('.textgrid'):
            path += '.TextGrid'
        grid = textgrid.TextGrid(path, strict=strict)
        grid.read(path)
    words = _load_tier(grid, word_tier)
    phones = _load_tier(grid, phone_tier)
    errors = []
    out = []
    phones = list(phones)
    max_time = phones[0].minTime
    for word in words._fillInTheGaps(' '):
        word_phones = []
        while phones and phones[0].maxTime <= word.maxTime:
            word_phones.append(phones.pop(0))
        # update max_time
        if word_phones:
            max_time = word_phones[-1].maxTime
        # resolve misaligned phones
        if max_time < word.maxTime and phones:
            start_dist = abs(phones[0].minTime - word.maxTime)
            stop_dist = abs(phones[0].maxTime - word.maxTime)
            if stop_dist < start_dist:
                word_phones.append(phones.pop(0))
                max_time = word_phones[-1].maxTime

        if not word_phones:
            continue
        word_pronunciation = tuple([phone.mark for phone in word_phones])
        word_times = tuple([phone.minTime for phone in word_phones])
        if word.mark in SILENCE:
            if not all(p in SILENCE for p in word_pronunciation):
                errors.append(f"{word.minTime:.3f}: Silence word tag {word.mark!r} but non-silent phones {word_pronunciation!r}")
            out.append(Realization((' ',), word_times[:1], ' ', max_time))
        else:
            if any(p in SILENCE for p in word_pronunciation):
                errors.append(f"{word.minTime:.3f}: Non-silence word tag {word.mark!r} includes silence phones {word_pronunciation!r}")
            out.append(Realization(word_pronunciation, word_times, word.mark, max_time))
    if errors:
        raise TextGridError('\n'.join(errors))

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
