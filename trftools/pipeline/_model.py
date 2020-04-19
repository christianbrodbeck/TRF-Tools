# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Model specification for TRFs

Adding a regressor
------------------

Questions:

 - Is there a brain region that responds to ``a``?
 - Is there a brain region that responds to ``a`` after controlling for ``b``?


-----   -------------   ---------------------
x1      x0              desc
-----   -------------   ---------------------
a + b   a + b$rnd       a + (b > $rnd)
b + a   b + a$rnd       b + (a > $rnd)
a + b   a$rnd + b$rnd   a + b > a$rnd + b$rnd
-----   -------------   ---------------------


Localization difference
-----------------------

Question:

 - Is there a brain region that represents ``a`` more than ``b``? E.g., where
   does categorical representation become more important than acoustic
   information?


Test 1:  ``a > b``

Correct for DFs and temporal characteristics of the regressors:

(a + b) - (a + b$rnd) > (a + b) - (a$rnd + b)

which is equivalent to

a + b$rnd > a$rnd + b
a + b$rnd > b + a$rnd


"""
from collections.abc import Sequence
from dataclasses import dataclass
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
from eelbrain._experiment.mne_experiment import DefinitionError
from eelbrain._utils import LazyProperty
from pyparsing import ParseException, Literal, Optional, Word, alphas, alphanums, delimitedList, nums, oneOf


COMP = {1: '>', 0: '=', -1: '<'}
TAIL = {'>': 1, '=': 0, '<': -1}


@dataclass(frozen=True)
class ModelTerm:
    stimulus: str
    code: str
    shuffle_index: Union[int, slice] = None
    shuffle: str = None
    shuffle_angle: int = 180

    @LazyProperty
    def string(self) -> str:
        items = [self.code]
        if self.stimulus:
            items.insert(0, f'{self.stimulus}|')
        if self.shuffle:
            items.append(self.shuffle_string)
        return ''.join(items)

    @LazyProperty
    def shuffle_string(self) -> str:
        if not self.shuffle:
            return ''
        items = ['$']
        if isinstance(self.shuffle_index, slice):
            items.append(f'[{self.shuffle_index.start}-{self.shuffle_index.stop}]')
        elif self.shuffle_index:
            items.append(f'[{self.shuffle_index}]')
        items.append(self.shuffle)
        if self.shuffle_angle != 180:
            items.append(str(self.shuffle_angle))
        return ''.join(items)


    @LazyProperty
    def without_shuffle(self):
        if self.shuffle:
            return ModelTerm(self.stimulus, self.code)
        else:
            return self

    def with_shuffle(self, index, shuffle, angle):
        if shuffle is None:
            return self.without_shuffle
        return ModelTerm(self.stimulus, self.code, index, shuffle, angle)

    def __repr__(self):
        return f"<ModelTerm: {self.string}>"


@dataclass(frozen=True)
class Model:
    """Model that can be fit to data"""
    terms: Tuple[ModelTerm, ...]

    @LazyProperty
    def name(self):
        return ' + '.join(term.string for term in self.terms)

    @LazyProperty
    def sorted(self):
        return '+'.join(sorted(term.string for term in self.terms))

    @classmethod
    def from_string(cls, string):
        try:
            return model.parseString(string, True)[0]
        except ParseException:
            raise DefinitionError(f"{string!r}: invalid Model")

    def __repr__(self):
        return f"<Model: {self.name}>"

    def __len__(self):
        return len(self.terms)

    def __add__(self, other: 'Model') -> 'Model':
        assert not self.intersection(other)
        return Model(self.terms + other.terms)

    def __sub__(self, other: 'Model') -> 'Model':
        if not all(term in self.terms for term in other.terms):
            missing = [term.string for term in other.terms if term not in self.terms]
            raise ValueError(f"Missing terms: {', '.join(missing)}")
        return Model(tuple([term for term in self.terms if term not in other.terms]))

    @classmethod
    def coerce(cls, x: Union['Model', str]):
        if isinstance(x, cls):
            return x
        else:
            return cls.from_string(x)

    @LazyProperty
    def has_randomization(self):
        return any(term.shuffle for term in self.terms)

    @LazyProperty
    def sorted_randomized(self):
        return '+'.join(sorted(term.string for term in self.terms if term.shuffle))

    @LazyProperty
    def terms_without_randomization(self):
        if self.has_randomization:
            return tuple([term.without_shuffle for term in self.terms])
        else:
            return self.terms

    @LazyProperty
    def without_randomization(self) -> 'Model':
        if self.has_randomization:
            return Model(self.terms_without_randomization)
        return self

    def difference(self, other: 'Model') -> 'Model':
        terms = [term for term in self.terms is term not in other.terms]
        return Model(tuple(terms))

    def intersection(self, other: 'Model') -> 'Model':
        terms = [term for term in self.terms if term in other.terms]
        return Model(tuple(terms))

    def randomized_component(self) -> 'Model':
        terms = [term for term in self.terms if term.shuffle]
        return Model(tuple(terms))

    def initialize(self, named_models: Dict[str, 'StructuredModel']) -> 'Model':
        terms = []
        for term in self.terms:
            if term.without_shuffle.string in named_models:
                model = named_models[term.without_shuffle.string].model
                if term.shuffle:
                    model = model.with_shuffle(term.shuffle_index, term.shuffle, term.shuffle_angle)
                terms.extend(model.terms)
            else:
                terms.append(term)
        return Model(tuple(terms))

    def multiple_permutations(self, n: int) -> List['Model']:
        """Generate multiple models with different shuffle angles"""
        if not self.has_randomization:
            raise TypeError(f"permutations={n} for model without randomization: {self.name}")
        elif not isinstance(n, int):
            raise TypeError(f"permutations={n}, need int")
        elif n <= 1:
            raise ValueError(f"permutations={n}")
        elif n >= 128:
            raise NotImplementedError(f"permutations={n}")
        n_to_go = n
        cycle = 2
        angles = []
        while n_to_go > 0:
            start = 360 / cycle
            new = np.arange(start, 360, start * 2)
            angles.extend(int(round(i)) for i in new[:n_to_go])
            n_to_go -= len(new)
            cycle *= 2
        return [self.with_angle(angle) for angle in angles]

    def with_shuffle(self, index, shuffle, angle) -> 'Model':
        """Apply shuffle settings to all terms"""
        if self.has_randomization:
            raise RuntimeError("Model already shuffled")
        terms = [term.with_shuffle(index, shuffle, angle) for term in self.terms]
        return Model(tuple(terms))

    def with_shuffled(self, term_to_shuffle: 'Term') -> 'Model':
        """Replace one term with a shuffled counterpart"""
        terms = list(self.terms)
        names = [term.string for term in terms]
        if term_to_shuffle.string not in names:
            raise ValueError(f"{term_to_shuffle.string}: not in {self.name}")
        index = names.index(term_to_shuffle.string)
        terms[index] = term_to_shuffle._model_term_with_shuffle()
        return Model(tuple(terms))

    def with_angle(self, angle: int) -> 'Model':
        """Apply shuffle angle to all shuffled terms"""
        terms = [term.with_shuffle(term.shuffle_index, term.shuffle, angle) for term in self.terms]
        return Model(tuple(terms))

    def without(self, term: str) -> 'Model':
        terms = list(self.terms)
        names = [term.string for term in terms]
        if term not in names:
            raise ValueError(f"{term}: not in {self.name}")
        del terms[names.index(term)]
        return Model(tuple(terms))


@dataclass(frozen=True)
class Term:
    "Term in StructuredModel"
    string: str
    parent: int = -1
    shuffle: str = 'shift'

    @classmethod
    def _coerce(cls, x, parent=-1, shuffle='shift'):
        if isinstance(x, cls):
            return x
        elif isinstance(x, str):
            return cls(x, parent, shuffle.lstrip('$'))
        elif isinstance(x, tuple):
            return cls._coerce(*x)
        raise TypeError(x)

    @LazyProperty
    def _model_term(self):
        return ModelTerm('', self.string)

    def _model_term_with_shuffle(self, index=None, shuffle=None, angle=180):
        return ModelTerm('', self.string, index, shuffle or self.shuffle, angle)


@dataclass(frozen=True)
class StructuredModel:
    "Model including information about each Term"
    terms: Tuple[Term]
    public_name: str = None

    @classmethod
    def coerce(cls, x):
        if isinstance(x, cls):
            return x
        elif isinstance(x, str):
            model = parse_model(x)
            if model.has_randomization:
                raise NotImplementedError(f"{x}: model with randomization")
            terms = [Term(term.string) for term in model.terms]
        elif isinstance(x, dict):
            terms = []
            for key, v in x.items():
                if isinstance(v, tuple):
                    term = Term._coerce(key, *v)
                elif isinstance(v, int):
                    term = Term(key, v)
                elif isinstance(v, str):
                    term = Term(key, shuffle=v)
                else:
                    raise DefinitionError(f"{x}: invalid term ({key})")
                terms.append(term)
        elif isinstance(x, Sequence):
            terms = [Term._coerce(term) for term in x]
        else:
            raise TypeError(x)
        return cls(tuple(terms))

    @LazyProperty
    def model(self):
        return Model(tuple([term._model_term for term in self.terms]))

    @LazyProperty
    def top_level_terms(self) -> List[Term]:
        parents = {term.parent for term in self.terms}
        return [term for i, term in enumerate(self.terms) if i >= -1 and i not in parents]

    def comparisons(self, cv: bool) -> List['Comparison']:
        if cv:
            return [Comparison(self.model, self.model.without(term.string), 1, f'{self.public_name} | {term.string}') for term in self.top_level_terms]
        else:
            return [Comparison(self.model, self.model.with_shuffled(term), 1, f'{self.public_name} | {term.string}${term.shuffle}') for term in self.top_level_terms]

    def without(self, term_to_remove: str):
        """Reduced model, excluding ``term``"""
        parents = {term.parent for term in self.terms}
        terms = []
        removed = False
        for i, term in enumerate(self.terms):
            if term.string == term_to_remove:
                if i in parents:
                    raise ValueError(f"{term_to_remove!r}: trying to remove parent term")
                removed = True
                continue
            elif removed and term.parent >= 0:
                term = Term(term.string, term.parent - 1, term.shuffle)
            terms.append(term)
        if not removed:
            raise ValueError(f"{term_to_remove!r}: not in model")
        return StructuredModel(tuple(terms))


@dataclass
class ComparisonSpec:
    x: Model

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> Union['Comparison', 'StructuredModel']:
        raise NotImplementedError


@dataclass
class TermComparisons(ComparisonSpec):

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> 'StructuredModel':
        assert not self.x.has_randomization
        x = self.x.initialize(named_models)
        # find terms to test
        terms = []
        for term in self.x.terms:
            if term.string in named_models:
                s_model = named_models[term.string]
                i = len(terms)
                for term in s_model.terms:
                    if i and term.parent >= 0:
                        term = Term(term.string, term.parent + i, term.shuffle)
                    terms.append(term)
            else:
                terms.append(Term(term.string))
        return StructuredModel(tuple(terms), self.x.name)


@dataclass
class DirectComparison(ComparisonSpec):
    operator: str
    x0: Model

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> 'Comparison':
        public_name = f"{self.x.name} {self.operator} {self.x0.name}"
        x = self.x.initialize(named_models)
        x0 = self.x0.initialize(named_models)
        tail = TAIL[self.operator]
        return Comparison(x, x0, tail, public_name)


@dataclass
class OmitComparison(ComparisonSpec):
    x_omit: Model

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> 'Comparison':
        public_name = f"{self.x.name} | {self.x_omit.name}"
        x = self.x.initialize(named_models)
        x_omit = self.x_omit.initialize(named_models)
        if x_omit.has_randomization:
            assert not cv
            x0 = x - x_omit.without_randomization + x_omit
        else:
            assert cv
            x0 = x - x_omit
        return Comparison(x, x0, 1, public_name)


@dataclass
class AddComparison(ComparisonSpec):
    x_add: Model

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> 'Comparison':
        public_name = f"{self.x.name} +| {self.x_add.name}"
        x = self.x.initialize(named_models)
        x_add = self.x_add.initialize(named_models)
        if x_add.has_randomization:
            assert not cv
            x1 = x + x_add.without_randomization
            x0 = x + x_add
        else:
            assert cv
            x1 = x + x_add
            x0 = x
        return Comparison(x1, x0, 1, public_name)


@dataclass
class Add2Comparison(ComparisonSpec):
    x1_add: Model
    operator: str
    x0_add: Model

    def initialize(
            self,
            named_models: Dict[str, StructuredModel],
            cv: bool = True,  # cross-validation (ignore shuffle)
    ) -> 'Comparison':
        public_name = f"{self.x.name} +| {self.x1_add.name} {self.operator} {self.x0_add.name}"
        x = self.x.initialize(named_models)
        x1_add = self.x1_add.initialize(named_models)
        x0_add = self.x0_add.initialize(named_models)
        x1 = x + x1_add
        x0 = x + x0_add
        return Comparison(x1, x0, TAIL[self.operator], public_name)


@dataclass(frozen=True)
class Comparison:
    """Model comparison for test or report"""
    x1: Model
    x0: Model
    tail: int = 1
    public_name: str = None

    @LazyProperty
    def models(self):
        return self.x1, self.x0

    @LazyProperty
    def common_base(self):
        return self.x1.intersection(self.x0)

    @LazyProperty
    def x1_only(self):
        return self.x1.difference(self.x0)

    @LazyProperty
    def x0_only(self):
        return self.x0.difference(self.x1)

    @LazyProperty
    def test_term_name(self):
        if not self.x0_only or self.x0_only.without_randomization == self.x1_only:
            return self.x1_only.name

    @LazyProperty
    def baseline_term_name(self):
        if len(self.x0_only) == 1 and self.x0_only.without_randomization == self.x1_only:
            return self.x0_only.name

    @LazyProperty
    def name(self):
        if self.public_name:
            return self.public_name
        op = COMP[self.tail]
        if self.common_base:
            if self.x1_only:
                if self.x0_only:
                    return f"{self.common_base} ({self.x1_only} {op} {self.x0_only}"
                else:
                    return f"{self.common_base} +| {self.x1_only}"
            elif self.x0_only:
                return f"{self.common_base} -| {self.x1_only}"
            else:
                raise RuntimeError("Comparing identical models")
        elif self.x1_only:
            if self.x0_only:
                return f"{self.x1_only} {op} {self.x0_only}"
            else:
                return f"{self.x1_only} {op} 0"
        elif self.x0_only:
            return f"0 {op} {self.x1_only}"
        raise RuntimeError("Comparing identical models")

    @classmethod
    def coerce(cls, x, cv=True, named_models={}):
        if isinstance(x, (cls, StructuredModel)):
            return x
        comp = parse_comparison(x)
        return comp.initialize(named_models, cv)

    def __repr__(self):
        return f"<Comparison: {self.name}>"


# components
integer = Word(nums).addParseAction(lambda s,l,t: int(t[0]))
pyword = Word(alphas+'_', alphanums+'_')

# shuffling
dash = Literal('-').suppress()
index_slice = integer + Optional(dash + integer, None)
index_slice.addParseAction(lambda s,l,t: t[0] if t[1] is None else slice(*t))
shuffle_index = Literal('[').suppress() + (index_slice ^ pyword) + Literal(']').suppress()
shuffle_method = oneOf('shift remask permute')
shuffle_suffix = Literal('$').suppress() + Optional(shuffle_index, None) + shuffle_method + Optional(integer, 180)

# term
stimulus_prefix = pyword + Literal('|').suppress().leaveWhitespace()
term_code = Word(alphas+'_', alphanums+'_-+*')
term = Optional(stimulus_prefix, '') + term_code + Optional(shuffle_suffix)
term.addParseAction(lambda s,l,t: ModelTerm(*t))

# model
model = delimitedList(term, '+').addParseAction(lambda s,l,t: Model(tuple(t)))

# comparison
term_comparisons = model.copy().addParseAction(lambda s,l,t: TermComparisons(*t))
direct_comparison = model + oneOf('= < >') + model
direct_comparison.addParseAction(lambda s,l,t: DirectComparison(*t))
omit_comparison = model + Literal('|').suppress() + model
omit_comparison.addParseAction(lambda s,l,t: OmitComparison(*t))
add_comparison = model + Literal('+|').suppress() + model
add_comparison.addParseAction(lambda s,l,t: AddComparison(*t))
add2_comparison = model + Literal('+|').suppress() + direct_comparison
add2_comparison.addParseAction(lambda s,l,t: Add2Comparison(t[0], t[1].x, t[1].operator, t[1].x0))
comparison = direct_comparison ^ omit_comparison ^ term_comparisons ^ add_comparison ^ add2_comparison


def parse_model(string: str) -> Model:
    try:
        parse = model.parseString(string, True)
    except ParseException:
        raise DefinitionError(f"{string!r}: invalid model")
    return parse[0]


def parse_comparison(string: str) -> ComparisonSpec:
    try:
        parse = comparison.parseString(string, True)
    except ParseException:
        raise DefinitionError(f"{string!r}: invalid comparison")
    return parse[0]


def save_models(models, path):
    out = [(k, v.name) for k, v in models.items()]
    with open(path, 'wb') as fid:
        pickle.dump(out, fid, pickle.HIGHEST_PROTOCOL)


def load_models(path):
    with open(path, 'rb') as fid:
        out = pickle.load(fid)
    return {k: parse_model(v) for k, v in out}


ModelArg = Union[Model, str]
