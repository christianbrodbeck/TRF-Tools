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
from itertools import chain,zip_longest
from os.path import commonprefix
import pickle
import re
from types import GeneratorType
from typing import Union, List

import numpy as np
from eelbrain._utils import LazyProperty


COMP = {1: '>', 0: '=', -1: '<'}
TAIL = {'>': 1, '=': 0, '<': -1}
BASE_RE = re.compile(r'^(.*)\s+(\+?\|)\s+(.*)$')
COMPARISON_RE = re.compile('(.* )(>|<|=)( .*)')
TERM_RE = re.compile(r'^ *([\[\]\w\d\-:|$]+) *$')

# characters that delineate randomization in a predictor
MOD_DELIM_CHARS = ('$',)


class ModelSyntaxError(Exception):
    "Error in model specification"
    def __init__(self, string, error):
        Exception.__init__(self, "%s (%s)" % (string, error))


def is_comparison(x):
    "Test whether str ``x`` is a comparison description (vs a model)"
    return bool(BASE_RE.match(x) or COMPARISON_RE.match(x))


class Model:
    """Model that can be fit to data"""
    def __init__(self, x):
        if isinstance(x, str):
            terms = x.split('+')
        elif isinstance(x, GeneratorType):
            terms = list(x)
        elif isinstance(x, Sequence):
            terms = x
        else:
            raise TypeError("x=%r" % (x,))

        matches = tuple(map(TERM_RE.match, terms))
        if not all(matches):
            invalid = (term for term, m in zip(terms, matches) if not m)
            raise ValueError(f"Invalid terms for model: {', '.join(map(repr, invalid))}")

        self.terms = tuple(m.group(1) for m in matches)
        self.name = ' + '.join(self.terms)
        self.sorted = '+'.join(sorted(self.terms))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Model) and self.name == other.name

    def __sub__(self, other: 'Model'):
        if not all(term in self.terms for term in other.terms):
            raise ValueError(f"Missing terms: {', '.join(term for term in other.terms if term not in self.terms)}")
        return Model([term for term in self.terms if term not in other.terms])

    @classmethod
    def coerce(cls, x, named_models={}):
        """Model that can be fit to data

        Notes
        -----
        Models can be fully specified (terms joined by '+') but can also be headed
        by a named model::

            model + y

        """
        if isinstance(x, cls):
            return x
        else:
            name_x1, terms = _model_terms(x, named_models)
            return cls(terms)

    @LazyProperty
    def has_randomization(self):
        return any('$' in term for term in self.terms)

    @LazyProperty
    def sorted_without_randomization(self):
        return '+'.join(sorted(self.terms_without_randomization))

    @LazyProperty
    def sorted_randomized(self):
        return '+'.join(sorted(term for term in self.terms if '$' in term))

    @LazyProperty
    def terms_without_randomization(self):
        if self.has_randomization:
            return tuple(term.partition('$')[0] for term in self.terms)
        else:
            return self.terms

    def __repr__(self):
        return "Model(%r)" % self.name

    def without_randomization(self):
        if self.has_randomization:
            return Model(self.terms_without_randomization)
        return self

    def randomized_component(self):
        return Model(term for term in self.terms if '$' in term)

    def multiple_permutations(self, n):
        """Generate multiple models with different permutations"""
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
        return (self.with_angle(angle) for angle in angles)

    def randomize(self, x, rand):
        terms = list(self.terms)
        if rand.startswith('$'):
            terms[terms.index(x)] += rand
        else:
            raise ValueError(f"rand={rand!r}")
        return Model(terms)

    def with_angle(self, angle):
        return Model(self._add_angle(t, angle) for t in self.terms)

    @staticmethod
    def _add_angle(term, angle):
        if angle == 180 or '$' not in term:
            return term
        head, tail = term.split('$')
        if re.match(r'\w+\d+', tail):
            raise ValueError(f"{term}: term alread has angle")
        return f'{head}${tail}{angle}'


class ComparisonBase:
    _COMP_ATTRS = ()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and all(getattr(self, attr) == getattr(other, attr) for attr in self._COMP_ATTRS)

    def __hash__(self):
        return hash((self.__class__, *(getattr(self, attr) for attr in self._COMP_ATTRS)))


class Comparison(ComparisonBase):
    """Model comparison for test or report
    
    Notes
    -----
    Different kinds of names:
     
     - readable: full test model, then the modification of the baseline model,
       to improve sorting.
     - concise: using the lest amount of space by extracting as much shared 
       information from the models as possible.
     - unique: insensitive to the order in which individual regressors are 
       specified.

    Attributes
    ----------
    x1_only : str
        Terms only in ``x1`` joined by ``+``.
    x0_only : str
        Terms only in ``x0`` joined by ``+``.
    """
    _COMP_ATTRS = ('x1', 'x0', 'tail')

    # TODO: detect phone-level2>3, feature-fea#manner
    # TODO: permute rows word-class$[c]shift or word-class[c]shift
    def __init__(self, x1, x0, tail=1, named_components={}, test_term_name=None):
        if tail is None:
            tail = 1
        self.x1 = x1 = Model.coerce(x1)
        self.x0 = x0 = Model.coerce(x0)
        self.tail = tail
        self.models = (x1, x0)
        self._components = {'x1': x1, **named_components}

        # preprocessing
        common_base = [t for t in x1.terms if t in x0.terms]
        x1_only = [t for t in x1.terms if t not in common_base]
        x0_only = [t for t in x0.terms if t not in common_base]
        randomized_terms = [t.split('$')[0] for t in x0.terms if '$' in t]
        x1_shuffled_in_x0 = [t for t in randomized_terms if t in x1.terms]
        if 'x0' not in self._components and not x0.has_randomization and not common_base:
            self._components['x0'] = x0  # -> show as "model1 = model0"

        x1_desc_terms = _model_desc_template(x1.without_randomization(), 'x1', self._components)
        if x1.has_randomization:
            # model | x$rand = y$rand
            x_base = x1.without_randomization()
            if x_base != x0.without_randomization():
                raise ValueError(f"x0 has term not in x1")
            x_base_desc = ' + '.join(x1_desc_terms)
            x1_desc = x1.randomized_component().name
            x0_desc = x0.randomized_component().name
            self._desc_template = f'{x_base_desc} | {x1_desc} {COMP[tail]} {x0_desc}'
        elif 'x0' in self._components:
            # model1 = model0
            x0_desc_terms = _model_desc_template(x0, 'x0', self._components)
            x1_desc = ' + '.join(x1_desc_terms)
            x0_desc = ' + '.join(x0_desc_terms)
            self._desc_template = '%s %s %s' % (x1_desc, COMP[tail], x0_desc)
        elif len(x1_shuffled_in_x0) == len(x0_only) == len(x1_only):
            # model | x$rand
            assert tail == 1
            if 'x0rand' in self._components:
                x0_desc_terms = _model_desc_template(Model(x0_only), 'x0rand', self._components)
                x0_desc = ' + '.join(x0_desc_terms)
            else:
                x0_desc = ' + '.join(x0_only)

            if x1_shuffled_in_x0 == x1_desc_terms[1:]:
                x1_desc = x1_desc_terms[0]
                op = '+|'
            else:
                x1_desc = ' + '.join(x1_desc_terms)
                op = '|'

            self._desc_template = f'{x1_desc} {op} {x0_desc}'
        else:
            # model | x = y
            x1_desc = ' + '.join(x1_desc_terms)
            self._desc_template = '%s | %s %s %s' % (x1_desc, ' + '.join(x1_only), COMP[tail], ' + '.join(x0_only))

        # term that is unique to the test model
        self.test_term_name = test_term_name or ' + '.join(x1_only)
        # term that the test-term is compared to
        self.baseline_term_name = None
        if test_term_name or len(x1_only) == 1:
            prefix = f'{self.test_term_name}$'
            for term in x0_only:
                if term.startswith(prefix):
                    self.baseline_term_name = term
                    break

        # concise name
        self.name, self.common_base, self.x1_only, self.x0_only = self.model_name(common_base, x1_only, x0_only, tail)

    def relative_name(self, model_names):
        """Generate name compatible with named models

        Parameters
        ----------
        model_names : {str: str}
            Names for models in ``self._components``.

        Notes
        -----
        Kinds of relative names: see :func:`parse_comparison`.
        """
        return self._desc_template.format_map(model_names)

    @staticmethod
    def model_name(common_base, x1_only, x0_only, tail):
        # modifies args in-place!
        if x1_only and x0_only:
            common_prefix = commonprefix(x1_only + x0_only)
        elif (common_base and (x1_only or x0_only) and all(
                item.startswith(common_base[-1]) for item in x1_only + x0_only)):
            common_prefix = common_base.pop(-1)
            x1_only.insert(0, common_prefix)
            x0_only.insert(0, common_prefix)
        else:
            common_prefix = ''

        if common_prefix:
            common_base.append(common_prefix)
            last_op = ' '
            prefix_len = len(common_prefix)
            x1_only = [i[prefix_len:] for i in x1_only]
            x0_only = [i[prefix_len:] for i in x0_only]
        else:
            last_op = ' + '

        common_base = ' + '.join(common_base)
        x1_only = ' + '.join(x1_only)
        x0_only = ' + '.join(x0_only).strip()
        name = "%s %s %s" % (x1_only, COMP[tail], x0_only)
        if common_base:
            name = "%s%s(%s)" % (common_base, last_op, name)
        return name, common_base, x1_only, x0_only

    @classmethod
    def coerce(cls, x1, x0=None, tail=None, named_models={}):
        if isinstance(x1, cls):
            assert x0 is None
            assert tail is None
            return x1
        if x1 in named_models:
            x1 = named_models[x1]
        if x0 in named_models:
            x0 = named_models[x0]
        elif x0 is None:
            assert isinstance(x1, str)
            assert tail is None
            return parse_comparison(x1, named_models)
        return cls(x1, x0, tail)

    def __repr__(self):
        args = (self.x1.name, self.x0.name)
        if self.tail != 1:
            args += (self.tail,)
        return "Comparison(%s)" % ', '.join(map(repr, args))


def _model_terms(string, named_models):
    """Find terms in ``model``, expanding any named models"""
    if isinstance(string, str):
        m = re.match(r'(.+) \((.+)\)', string)
        if m:
            a, b = m.groups()
            _, model_terms = _model_terms(a, named_models)
            _, randomized_terms = _model_terms(b, named_models)
            for x in randomized_terms:
                i = model_terms.index(x[:x.index('$')])
                model_terms[i] = x
        else:
            model_terms = map(str.strip, string.split(' + '))
    else:
        model_terms = string
    model_terms = list(model_terms)
    term_0 = model_terms[0]
    if '$' in term_0:
        term_0, rand = term_0.split('$')
    else:
        rand = None

    if named_models and term_0 in named_models:
        named_model = named_models[term_0]
        if rand:
            model_terms[:1] = ('%s$%s' % (term, rand) for term in named_model.terms)
        else:
            model_terms[:1] = named_model.terms
        return named_model, model_terms
    else:
        return None, model_terms


def _model_desc_template(model, name, named_models):
    if name not in named_models:
        return ['{%s}']
    named_model = named_models[name]
    if named_model.has_randomization:
        raise NotImplementedError("named_model with randomization")
    if model.has_randomization:
        named_terms = (tr for t, tr in zip(model.terms_without_randomization, model.terms) if t in named_model.terms)
        rand = {tr.partition('$')[2] for tr in named_terms}
        assert len(rand) == 1
        name_template = '{%s}$%s' % (name, rand.pop())
        unnamed_terms = [tr for t, tr in zip(model.terms_without_randomization, model.terms) if t not in named_model.terms]
    else:
        name_template = '{%s}' % name
        unnamed_terms = [t for t in model.terms if t not in named_model.terms]
    return [name_template, *unnamed_terms]


def parse_comparison(string, named_models=None, test_term_name=None):
    """Parse a comparison string (see TRFExperiment module docstring)"""
    named_components = {}
    n_parentheses = string.count('(')
    if n_parentheses != string.count(')'):
        raise ValueError(f"Unequal number of opening and closing parentheses in {string!r}")

    # prepended model: model | comparison
    m = BASE_RE.match(string)
    if m:
        if n_parentheses:
            raise NotImplementedError(f"Comparisons with | and parentheses: {string!r}")
        base_string, sep, comparison_string = m.groups()
        name_x1, base_terms = _model_terms(base_string, named_models)
        if name_x1:
            named_components['x1'] = name_x1
    else:
        comparison_string = string
        sep = base_terms = None

    # common base from parentheses
    if n_parentheses == 0:
        common_base = None
    elif n_parentheses == 1:
        i_open = comparison_string.index('(')
        i_close = comparison_string.index(')')
        if comparison_string[i_close + 1:].strip():
            raise NotImplementedError(f"Expression after closing parentheses in {string!r}")
        common_base = comparison_string[:i_open]
        comparison_string = comparison_string[i_open + 1: i_close]
    else:
        raise NotImplementedError(f"{string!r}: more than one set of parentheses")

    # comparison (</=/>)
    m = COMPARISON_RE.match(comparison_string)
    if m:
        assert sep != '+|'
        x1, comp, x0 = m.groups()
        tail = TAIL[comp]
        comp_name_x1, x1_only_terms = _model_terms(x1, named_models)
        name_x0, x0_only_terms = _model_terms(x0, named_models)
        if name_x0:
            named_components['x0'] = name_x0

        if base_terms:
            # model | x > y
            if comp_name_x1 or name_x0:
                raise ValueError(f"{string!r}: marginal term (after |) can't contain named model")
            elif any(t in x0_only_terms for t in x1_only_terms):
                raise ValueError(f"{string!r}: models right of | share terms")
            elif any(t in base_terms for t in x0_only_terms):
                raise ValueError(f"{string!r}: term right of {comp} in base model")

            if all(t in base_terms for t in x1_only_terms):
                # model | x = u
                x1_terms = [t for t in base_terms if t not in x0_only_terms]
                x1_terms.extend(t for t in x1_only_terms if t not in base_terms)
                x0_terms = [t for t in base_terms if t not in x1_only_terms]
                x0_terms.extend(t for t in x0_only_terms if t not in base_terms)
            elif all('$' in t for t in chain(x1_only_terms, x0_only_terms)):
                x1_terms = list(base_terms)
                for term in x1_only_terms:
                    x1_terms[x1_terms.index(term.split('$')[0])] = term
                x0_terms = list(base_terms)
                for term in x0_only_terms:
                    x0_terms[x0_terms.index(term.split('$')[0])] = term
            else:
                raise ValueError(f"{string!r}: invalid right of |")
        else:
            # x > y
            if comp_name_x1:
                named_components['x1'] = comp_name_x1
            x1_terms = x1_only_terms
            x0_terms = x0_only_terms
    elif base_terms:
        # model | x$rand
        tail = 1
        x1_terms = base_terms
        name_x0, right_terms = _model_terms(comparison_string, named_models)
        if not all('$' in t for t in right_terms):
            raise ValueError(f"{string!r}: no randomization right of the model")
        if name_x0:
            named_components['x0rand'] = name_x0
        rand_terms = {term.partition('$')[0]: term for term in right_terms}
        if sep == '+|':
            duplicate = set(x1_terms).intersection(rand_terms)
            if duplicate:
                raise ValueError(f"{string!r}: duplicate terms {', '.join(duplicate)}")
            x1_terms.extend(rand_terms)
        missing = set(rand_terms).difference(x1_terms)
        if missing:
            raise ValueError(f"{string!r}: terms after | that are not in base model: {', '.join(missing)}")
        x0_terms = [rand_terms.get(term, term) for term in base_terms]
    else:
        raise ValueError(f"{string!r}: Can't split {comparison_string!r} into two models")

    # add common base to models
    if common_base:
        name_x1, common_base_terms = _model_terms(common_base, named_models)
        if name_x1:
            raise NotImplementedError(f"Named model in comparison without |: {string!r}")
        prefix = common_base_terms.pop(-1)
        if prefix:
            x1_terms = [prefix + item for item in x1_terms]
            x0_terms = [prefix + item for item in x0_terms]
        else:
            if x1_terms == ['']:
                x1_terms = []
            if x0_terms == ['']:
                x0_terms = []
        x1_terms = common_base_terms + x1_terms
        x0_terms = common_base_terms + x0_terms

    return Comparison(x1_terms, x0_terms, tail, named_components, test_term_name)


class Comparisons(ComparisonBase):
    "Baseclass for comparison groups"
    _COMP_ATTRS = ('comparisons',)

    def __init__(self, name: str, comparisons: tuple):
        self.name = name
        self.comparisons = comparisons


class IncrementalComparisons(Comparisons):
    """Incremental comparison for each term in ``x``

    Parameters
    ----------
    x : Model
        The full model.
    rand : str | list of str
        Randomization method, general or per term.
    hierarchy : list of int
        Indicate for each term its parent (by index); used to find reduced
        models. ``-1`` are root terms, ``-2`` are terms that are not considered
        for exclusion. By default, all terms are considered.
    """
    def __init__(self, x: 'ModelArg', rand: str = '$shift', hierarchy: List[int] = None):
        x = Model.coerce(x)
        n_terms = len(x.terms)
        if isinstance(rand, str):
            rands = (rand,) * n_terms
            default_rand = rand == '$shift'
        else:
            rands = rand = tuple(rand)
            if len(rands) != n_terms:
                raise ValueError(f"rand={rand!r}; must have one entry per term ({n_terms} not {len(rand)}")
            default_rand = all(r == '$shift' for r in rands)

        # find terms which require a comparison
        if hierarchy is None:
            comp_terms = enumerate(rands)
        else:
            hierarchy = tuple(map(int, hierarchy))
            if len(hierarchy) != n_terms:
                raise ValueError(f"hierarchy={hierarchy}; must have one entry per term ({n_terms} not {len(hierarchy)})")
            elif not all(-2 <= i < n_terms for i in hierarchy):
                raise ValueError(f"hierarchy={hierarchy}: invalid values")
            comp_terms = []
            for i, parent in enumerate(hierarchy):
                if parent == -2:
                    continue
                elif i in hierarchy:
                    continue
                else:
                    comp_terms.append((i, rands[i]))

        comparisons = []
        for i, rand_str in comp_terms:
            if rand_str.startswith('$'):
                x0 = list(x.terms)
                x0[i] += rand_str
                comparison = Comparison(x, x0)
            elif rand_str.startswith('|'):
                comparison = parse_comparison(f"{x.name} {rand_str}", test_term_name=x.terms[i])
            else:
                raise ValueError(f"Randomization {rand_str!r} (needs to start with $ or |)")
            comparisons.append(comparison)
        Comparisons.__init__(self, x.name, tuple(comparisons))
        self._default_rand = default_rand
        # inherited attributes
        self.x = x
        self.terms = x.terms
        self.rand = rand
        self.hierarchy = hierarchy

    @classmethod
    def coerce(cls, x):
        if isinstance(x, cls):
            return x
        elif isinstance(x, dict):
            return IncrementalComparisons.from_effect_dict(x)
        else:
            return IncrementalComparisons(x)

    @classmethod
    def from_effect_dict(cls, effects: dict, rand: str = '$shift'):
        """Initialize from list of dict

        Parameters
        ----------
        effects : dict
            Each effect is coded as ``{term: parent}`` or
            ``{term: (parent, rand)}`` entry.
        rand : str
            Default randomization (applied to all terms that do not explicitly
            mention randomization).
        """
        lines = []
        for term, entry in effects.items():
            if isinstance(entry, int):
                parent, rand_ = entry, rand
            elif isinstance(entry, str):
                parent, rand_ = -1, entry
            else:
                parent, rand_ = entry
            lines.append([term, rand_, parent])
        return cls(*zip(*lines))

    @classmethod
    def from_effect_list(cls, effects):
        """Initialize from list of ``(code, hierarchy, randomization)``

        Parameters
        ----------

        Notes
        -----
        If hierarchy is -2, ``randomization`` can be omitted.
        """
        terms, hierarchy, rand = zip_longest(*effects)
        return cls(terms, rand, hierarchy)

    def __repr__(self):
        args = [repr(self.x.name)]
        if self.rand != '$shift':
            args.append("rand=%r" % (self.rand,))
        if self.hierarchy is not None:
            args.append("hierarchy=%r" % (self.hierarchy,))
        return "IncrementalComparisons(%s)" % ', '.join(args)

    def reduce(self, term):
        """Reduced model, excluding ``term``"""
        terms = list(self.x.terms)
        i = terms.index(term)
        del terms[i]
        if isinstance(self.rand, str):
            rand = self.rand
        else:
            rand = list(self.rand)
            del rand[i]
        if self.hierarchy is None:
            hierarchy = None
        elif i in self.hierarchy:
            raise ValueError(f"term={term!r}: trying to remove parent term")
        else:
            hierarchy = [h if h < i else h - 1 for h in self.hierarchy]
            del hierarchy[i]
        return IncrementalComparisons(terms, rand, hierarchy)


def save_models(models, path):
    out = [(k, v.name) for k, v in models.items()]
    with open(path, 'wb') as fid:
        pickle.dump(out, fid, pickle.HIGHEST_PROTOCOL)


def load_models(path):
    with open(path, 'rb') as fid:
        out = pickle.load(fid)
    return {k: Model(v) for k, v in out}


ModelArg = Union[Model, str]
