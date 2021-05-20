import enum

from eelbrain import fmtxt
from eelbrain._text import ms
from eelbrain._stats.test import star
from eelbrain._stats.testnd import MultiEffectNDTest
from eelbrain.testnd import LMGroup


class TestType(enum.Enum):
    DIFFERENCE = enum.auto()
    MULTI_EFFECT = enum.auto()
    TWO_STAGE = enum.auto()

    @classmethod
    def for_test(cls, test):
        if isinstance(test, LMGroup):
            return cls.TWO_STAGE
        elif isinstance(test, MultiEffectNDTest):
            return cls.MULTI_EFFECT
        else:
            return cls.DIFFERENCE


class ResultCollection(dict):
    test_type = None
    _statistic = None

    def __init__(self, tests: dict = ()):
        dict.__init__(self, tests)
        for key, test in self.items():
            self._validate_test(test)

    def _validate_test(self, test):
        test_type = TestType.for_test(test)
        if self.test_type is None:
            self.test_type = test_type
            if test_type is not TestType.TWO_STAGE:
                self._statistic = test._statistic
        elif test_type is not self.test_type:
            raise TypeError(f"{test}: all tests need to be of the same type ({self.test_type})")

    def __reduce__(self):
        return self.__class__, (dict(self),)

    def __setitem__(self, key, test):
        self._validate_test(test)
        dict.__setitem__(self, key, test)

    def __repr__(self):
        return f"<ResultCollection: {', '.join(self.keys())}>"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise NotImplementedError
        table = str(self.table()).splitlines()
        lines = ("<ResultCollection:", *(f'  {line}' for line in table), '>')
        p.text('\n'.join(lines))

    def _default_plot_obj(self):
        out = [test._default_plot_obj() for test in self.values()]
        if isinstance(out[0], list):
            return [y for ys in out for y in ys]
        return out

    def clusters(self, p=0.05):
        """Table with significant clusters"""
        if self.test_type is TestType.TWO_STAGE:
            raise NotImplementedError
        else:
            table = fmtxt.Table('lrrrrll')
            table.cells('Effect', 't-start', 't-stop', fmtxt.symbol(self._statistic, 'max'), fmtxt.symbol('t', 'peak'), fmtxt.symbol('p'), 'sig', just='l')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                table.endline()
                clusters = res.find_clusters(p, maps=True)
                clusters.sort('tstart')
                if self.test_type is not TestType.MULTI_EFFECT:
                    clusters[:, 'effect'] = ''
                for effect, tstart, tstop, p_, sig, cmap in clusters.zip('effect', 'tstart', 'tstop', 'p', 'sig', 'cluster'):
                    max_stat, max_time = res._max_statistic(mask=cmap != 0, return_time=True)
                    table.cells(f'  {effect}', ms(tstart), ms(tstop), fmtxt.stat(max_stat), ms(max_time), fmtxt.p(p_), sig)
        return table

    def table(self, title=None, caption=None):
        """Table with effects and smallest p-value"""
        if self.test_type is TestType.TWO_STAGE:
            cols = sorted({col for res in self.values() for col in res.column_names})
            table = fmtxt.Table('l' * (1 + len(cols)), title=title, caption=caption)
            table.cell('')
            table.cells(*cols)
            table.midrule()
            for key, lmg in self.items():
                table.cell(key)
                for res in (lmg.tests[c] for c in cols):
                    pmin = res.p.min()
                    table.cell(fmtxt.FMText([fmtxt.p(pmin), star(pmin)]))
        elif self.test_type is TestType.MULTI_EFFECT:
            table = fmtxt.Table('lllll', title=title, caption=caption)
            table.cells('Test', 'Effect', fmtxt.symbol(self._statistic, 'max'), fmtxt.symbol('p'), 'sig')
            table.midrule()
            for key, res in self.items():
                for i, effect in enumerate(res.effects):
                    table.cells(key, effect)
                    pmin = res.p[i].min()
                    table.cell(fmtxt.stat(res._max_statistic(i)))
                    table.cell(fmtxt.p(pmin))
                    table.cell(star(pmin))
                    key = ''
        else:
            table = fmtxt.Table('llll', title=title, caption=caption)
            table.cells('Effect', fmtxt.symbol(self._statistic, 'max'), fmtxt.symbol('p'), 'sig')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                pmin = res.p.min()
                table.cell(fmtxt.stat(res._max_statistic()))
                table.cell(fmtxt.p(pmin))
                table.cell(star(pmin))
        return table
