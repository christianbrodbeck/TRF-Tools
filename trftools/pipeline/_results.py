from eelbrain import fmtxt
from eelbrain._text import ms
from eelbrain._utils import LazyProperty
from eelbrain._stats.test import star
from eelbrain.testnd import LMGroup, anova


class ResultCollection(dict):

    def __reduce__(self):
        return self.__class__, (dict(self),)

    @LazyProperty
    def test_type(self):
        for res in self.values():
            return type(res)

    def __setitem__(self, key, test):
        if self.test_type is None:
            self.test_type = type(test)
        elif type(test) is not self.test_type:
            raise TypeError(f"{test}: all tests need to be of the same type ({self.test_type})")
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
        if self.test_type is LMGroup:
            raise NotImplementedError
        else:
            table = fmtxt.Table('lrrll')
            table.cells('Effect', 't-start', 't-stop', fmtxt.symbol('p'), 'sig', just='l')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                table.endline()
                clusters = res.find_clusters(p)
                clusters.sort('tstart')
                if self.test_type != anova:
                    clusters[:, 'effect'] = ''
                for effect, tstart, tstop, p_, sig in clusters.zip('effect', 'tstart', 'tstop', 'p', 'sig'):
                    table.cells(f'  {effect}', ms(tstart), ms(tstop), fmtxt.p(p_), sig)
        return table

    def table(self, title=None, caption=None):
        """Table with effects and smallest p-value"""
        if self.test_type is LMGroup:
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
        elif self.test_type is anova:
            table = fmtxt.Table('lllll', title=title, caption=caption)
            table.cells('Test', 'Effect', fmtxt.symbol(self.test_type._statistic, 'max'), fmtxt.symbol('p'), 'sig')
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
            table.cells('Effect', fmtxt.symbol(self.test_type._statistic, 'max'), fmtxt.symbol('p'), 'sig')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                pmin = res.p.min()
                table.cell(fmtxt.stat(res._max_statistic()))
                table.cell(fmtxt.p(pmin))
                table.cell(star(pmin))
        return table
