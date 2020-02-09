import random
import re

from eelbrain import Dataset
from eelbrain._experiment.definitions import CodeBase, CodeError
import numpy as np

from .._ndvar import SHUFFLE_METHODS as NDVAR_SHUFFLE_METHODS


VALUE_SHUFFLE_METHODS = ('permute', 'relocate')
SHUFFLE_METHODS = NDVAR_SHUFFLE_METHODS + VALUE_SHUFFLE_METHODS


class Code(CodeBase):
    """Create variable with '-' delimited instructions

    Attributes
    ----------
    string : str
        The complete code string.
    stim : str
        The stimulus; the part preceding ``|``.
    code : str
        The main body of the code; the part between ``|`` and ``$``.
    shuffle : str
        The shuffling method.
    has_permutation : bool
        The regressor is distorted in some way.
    has_randomization : bool
        The regressor is distorted in a way that includes randomness.
    """
    _sep = '-'
    _seed = None

    def __init__(self, string):
        m = re.match(r'(?:([\w+-]+)\|)?([\w:-]+)(?:\$(-?\d*-?)([a-zA-Z]+)(\d*))?$', string)
        if not m:
            raise CodeError(string, "not a valid code")
        stim, code_string, shuffle_band, shuffle, angle = m.groups()
        if shuffle:
            self.code_with_rand = f'{code_string}${shuffle_band}{shuffle}{angle}'
            if angle:
                angle = int(angle)
                if angle == 180:
                    raise CodeError(string, "shuffle angle '180' should be omitted")
                elif not 360 > angle > 0:
                    raise CodeError(string, f"shuffle angle {angle}")
            else:
                angle = 180

            if shuffle_band:
                m = re.match(r'^(-?)(\d+)(-?)$', shuffle_band)
                if not m:
                    raise ValueError(f'{string!r} (shuffle index)')
                pre, index, post = m.groups()
                if pre:
                    if post:
                        raise ValueError(f'{string!r} (shuffle index)')
                    shuffle_band = slice(int(index))
                elif post:
                    shuffle_band = slice(int(index), None)
                else:
                    shuffle_band = int(index)
            else:
                shuffle_band  = None
        else:
            self.code_with_rand = code_string
            shuffle_band = shuffle = angle = None
        self.stim = stim or None
        self.code = code_string
        self.shuffle = shuffle
        self.shuffle_band = shuffle_band
        self.shuffle_angle = angle
        CodeBase.__init__(self, string, code_string)
        self.has_randomization = shuffle in VALUE_SHUFFLE_METHODS or '>' in string
        self.has_permutation = shuffle in SHUFFLE_METHODS or '>' in string
        self._shuffle_done = False
        self.key = Dataset.as_key(self.string)

    def register_string_done(self):
        self._i = len(self._items) - 1

    def register_shuffle(self):
        "Register that shuffling has been done"
        if not self.shuffle:
            raise RuntimeError(f"Code does not require shuffling: {self.code}")
        elif self._shuffle_done:
            raise RuntimeError(f"Already shuffled: {self.code}")
        self._shuffle_done = True

    def assert_done(self):
        CodeBase.assert_done(self)
        if self.shuffle and not self._shuffle_done:
            raise self.error("Shuffling not performed", i=-1)

    def _get_rng(self):
        if self._seed is None:
            raise RuntimeError(f"{self} not seeded")
        return np.random.RandomState(self._seed)

    def seed(self, subject=None):
        "Seed random state"
        if not self.shuffle:
            return
        elif self._seed is not None:
            raise RuntimeError(f"{self} seeded twice")
        if subject is None:
            seed = 0
            angle_magnitude = 10
        else:
            digits = ''.join(re.findall(r'\d', subject))
            seed = int(digits)
            angle_magnitude = 10**len(digits)
        seed += self.shuffle_angle * angle_magnitude
        self._seed = seed

    @property
    def string_without_rand(self):
        if self.stim:
            return f'{self.stim}|{self.code}'
        else:
            return self.code

    def with_stim(self, stim):
        "Copy of the code with different stimulus"
        code_string = self.string
        if self.stim:
            i = len(self.stim) + 1
            code_string = code_string[i:]
        return Code(f'{stim}|{code_string}')


def parse_index(code):
    """Parse current item in ``code`` for an index expression

    Returns
    -------
    index : None | int | slice
        The index that was found.
    """
    item = code.lookahead()
    if not item:
        return
    m = re.match(r"^(\d?)(:?)(\d?)$", item)
    if not m:
        return
    code.next()
    start, colon, stop = m.groups()
    if start or stop:
        start = int(start) if start else None
        stop = int(stop) if stop else None
        if stop is not None and stop < (start or 0) + 2:
            raise code.error("Redundant slice definition (length 1)")
    else:
        raise code.error("Index does nothing")
    if colon:
        if start == 0:
            raise code.error("Redundant definition (omit '0' in '0:')")
        return slice(start, stop)
    else:
        assert stop is None
        return start
