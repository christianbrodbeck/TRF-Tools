from math import ceil
from typing import Any, Dict, Sequence, Tuple, Union

from eelbrain import plot, fmtxt, find_peaks, Dataset, resample
from eelbrain._stats.testnd import MultiEffectNDTest
from eelbrain._stats.spm import LMGroup
from eelbrain.fmtxt import FMTextArg, Figure, Section, FMText

from ._results import ResultCollection


class BrainLayout:
    _views = {
        'temporal': ((-18, -28, 50), 1.5),
    }

    def __init__(
            self,
            brain_view: Union[str, Tuple[float, ...]],
            axw: float = None,
    ):
        if isinstance(brain_view, str):
            brain_view, default_axw = self._views[brain_view]
        else:
            default_axw = 2.5
        if axw is None:
            axw = default_axw
        self.brain_view = brain_view
        self.axw = axw
        self.dpi = 144  # good for notebook
        self.table_args = {'dpi': self.dpi, 'axw': self.axw, 'show': False}


def ceil_vlim(ndvar):
    vlim = max(abs(ndvar.min()), ndvar.max())
    return ceil(vlim * 100) / 100


def vsource_tfce_result(res, title, desc):
    """Section for volume-source space TFCE result object"""
    content = [desc]
    if isinstance(res, (MultiEffectNDTest, LMGroup)):
        if isinstance(res, LMGroup):
            effects = res.column_names
            stat_maps = [res.tests[e].masked_parameter_map() for e in effects]
            max_stats = [res.tests[e]._max_statistic() for e in effects]
            ps = [test.p.min() for test in res.tests]
        else:
            effects = res.effects
            stat_maps = [res.masked_parameter_map(e) for e in effects]
            max_stats = [res._max_statistic(e) for e in effects]
            ps = [x.min() for x in res.p]

        for effect, stat_map, p, max_statistic in zip(effects, stat_maps, ps, max_stats):
            fig = vsource_tfce_map(effect.capitalize(), stat_map, p, res._statistic, max_statistic)
            content.append(fig)
        p = min(ps)
    else:
        p = res.p.min()
        fig = vsource_tfce_map("Increase of z-transformed correlation", res.masked_difference(), p, res._statistic, res._max_statistic())
        content.append(fig)

    title = FMText([title, '  (', fmtxt.peq(p), fmtxt.Stars.from_p(p, tag=None), ')'])
    section = Section(title, content)
    return section


def vsource_tfce_map(effect, stat_map, p, statistic, max_statistic):
    brain = plot.GlassBrain(stat_map, show=False)
    cbar = brain.plot_colorbar(orientation='vertical', show=False, width=0.2, w=1.1, h=2.8)
    content = [brain.image(), cbar.image()]
    brain.close()
    cbar.close()

    caption = FMText([f"{effect}; ", fmtxt.eq(statistic, max_statistic, 'max'), ', ', fmtxt.peq(p)])
    return Figure(content, caption)


def source_results(
        ress: ResultCollection,
        ress_hemi: ResultCollection = None,
        heading: FMTextArg = None,
        brain_view: Union[str, Sequence[float]] = None,
        axw: float = None,
        surf: str = 'inflated',
        cortex: Any = ((1.00,) * 3, (.4,) * 3),
        sig: bool = True,
        vmax: float = None,
        cmap: str = None,
        alpha: float = 1.,
):
    "Only used for TRFExperiment model-test"
    layout = BrainLayout(brain_view, axw)

    if heading is not None:
        doc = fmtxt.Section(heading)
    else:
        doc = fmtxt.FMText()

    tables = [ress.table(title='Model test')]
    if ress_hemi is not None:
        tables.append(ress_hemi.table(title="Lateralization"))
    doc.append(fmtxt.Figure(fmtxt.FloatingLayout(tables)))

    if sig and all(res.p.min() > 0.05 for res in ress.values()):
        return doc

    # plots tests
    panels = []
    all_ress = (ress,) if ress_hemi is None else (ress, ress_hemi)
    for ress_i in all_ress:
        sp = plot.brain.SequencePlotter()
        if layout.brain_view:
            sp.set_parallel_view(*layout.brain_view)
        sp.set_brain_args(surf=surf, cortex=cortex)
        for x, res in ress_i.items():
            y = res.masked_difference() if sig else res.difference
            sp.add_ndvar(y, label=x, cmap=cmap, vmax=vmax, alpha=alpha)
        panel = sp.plot_table(view='lateral', orientation='vertical', **layout.table_args)
        panels.append(panel)
    doc.append(fmtxt.Figure(panels))
    for panel in panels:
        panel.close()
    return doc


def source_tfce_result(res, surfer_kwargs, title, desc, brain=None):
    """Section for TFCE result object"""
    content = [desc]
    if isinstance(res, (MultiEffectNDTest, LMGroup)):
        if isinstance(res, LMGroup):
            effects = res.column_names
            p_maps = [res.tests[e].p for e in effects]
            stat_maps = [res.tests[e]._statistic_map for e in effects]
            max_stats = [res.tests[e]._max_statistic() for e in effects]
        else:
            effects = res.effects
            p_maps = res.p
            stat_maps = res.f
            max_stats = [f.max() for f in res.f]
        p = 1.
        for effect, stat_map, p_map, max_statistic in zip(effects, stat_maps, p_maps, max_stats):
            p = min(p, p_map.min())
            fig, brain = source_tfce_pmap(effect.capitalize(), p_map, stat_map, max_statistic, surfer_kwargs, brain)
            content.append(fig)
    else:
        p = res.p.min()
        fig, brain = source_tfce_pmap("Increase of z-transformed correlation", res.p, res._statistic_map, res._max_statistic(), surfer_kwargs, brain, res.difference)
        content.append(fig)

    title = FMText([title, '  (', fmtxt.peq(p), fmtxt.Stars.from_p(p, tag=None), ')'])
    section = Section(title, content)
    return section, brain


def source_tfce_pmap(effect, pmap, statmap, max_statistic, surfer_kwargs, brain, difference=None):
    if brain is None:
        brain = plot.brain.brain(pmap.source, **surfer_kwargs)
    else:
        brain.remove_data()

    brain.add_ndvar_p_map(pmap, statmap)
    cbar = brain.plot_colorbar(orientation='vertical', show=False, width=0.2, w=1.1, h=2.8)
    content = [brain.image(), cbar.image()]
    cbar.close()

    if difference is not None:
        brain.remove_data()
        brain.add_ndvar(difference)
        cbar = brain.plot_colorbar(orientation='vertical', show=False, width=0.2, w=1.1, h=2.8)
        content.append(brain.image())
        content.append(cbar.image())
        cbar.close()

    statistic = statmap.info['meas']
    p = pmap.min()
    caption = FMText([f"{effect}; ", fmtxt.eq(statistic, max_statistic, 'max'), ', ', fmtxt.peq(p)])
    fig = Figure(content, caption)
    return fig, brain


def find_peak_times(y, y_mask):
    y = y.smooth('time', 0.15)
    peak_t = find_peaks(y).nonzero()[0]
    peak_v = [y[t] for t in peak_t]
    baseline_v = y[0]
    return [t for i, t in enumerate(peak_t) if peak_v[i] > baseline_v and y_mask.sub(time=t)]


def source_trfs(
        ress: ResultCollection,
        heading: FMTextArg = None,
        brain_view: Union[str, Sequence[float]] = None,
        axw: float = None,
        surf: str = 'inflated',
        cortex: Any = ((1.00,) * 3, (.4,) * 3),
        vmax: float = None,
        xlim: Tuple[float, float] = None,
        times: Sequence[float] = None,
        cmap: str = None,
        labels: Dict[str, str] = None,
        rasterize: bool = None,
        brain_timewindow: float = 0.050
):
    "Only used for TRFExperiment model-test"
    layout = BrainLayout(brain_view, axw)
    dt = brain_timewindow / 2

    if heading is not None:
        doc = fmtxt.Section(heading)
    else:
        doc = fmtxt.FMText()

    if cmap is None:
        cmap = 'lux-a'

    if labels is None:
        labels = {}

    trf_table = fmtxt.Table('ll')
    for key, res in ress.items():
        trf_resampled = resample(res.masked_difference(), 1000)
        label = labels.get(key, key)
        if rasterize is None:
            rasterize = len(trf_resampled.source) > 500
        # times for anatomical plots
        if times is None:
            trf_tc = abs(trf_resampled).sum('source')
            trf_tc_mask = (~trf_resampled.get_mask()).sum('source') >= 10
            times_ = find_peak_times(trf_tc, trf_tc_mask)
        else:
            times_ = times
        # butterfly-plot
        p = plot.Butterfly(trf_resampled, h=3, w=4, ylabel=False, title=label, vmax=vmax, xlim=xlim, show=False)
        for t in times_:
            p.add_vline(t, color='k')
        trf_table.cell(fmtxt.asfmtext(p, rasterize=rasterize))
        p.close()
        # peak sources
        if not times_:
            trf_table.cell()
        sp = plot.brain.SequencePlotter()
        if layout.brain_view:
            sp.set_parallel_view(*layout.brain_view)
        sp.set_brain_args(surf=surf, cortex=cortex)
        for t in times_:
            yt = trf_resampled.mean(time=(t - dt, t + dt + 0.001))
            if isinstance(cmap, str):
                vmax_ = vmax or max(-yt.min(), yt.max()) or 1
                cmap_ = plot.soft_threshold_colormap(cmap, vmax_ / 10, vmax_)
            else:
                cmap_ = cmap
            sp.add_ndvar(yt, cmap=cmap_, label=f'{t * 1000:.0f} ms', smoothing_steps=10)
        p = sp.plot_table(view='lateral', orientation='vertical', **layout.table_args)
        trf_table.cell(p)
        p.close()
    doc.append(fmtxt.Figure(trf_table))
    return doc


def source_trfs_in_timebins(report, ds, surfer_kwargs, tstep=0.05):
    """Add comparison of two models to the report"""
    # TRFs
    sec = report.add_section("TRFs")
    for name in ds.info['xs']:
        subsec = sec.add_section(name)
        trf = ds[name].mean('case')
        if trf.ndim == 3:
            dim, _, _ = trf.get_dimnames((None, 'time', 'source'))
            trf = trf.rms(dim)
            caption = "RMS of the TRF of %s."
        elif trf.ndim == 2:
            caption = "TRF of %s."
        else:
            raise NotImplementedError("TRF with ndim=%i: %s" % (trf.ndim, trf))
        vlim = ceil_vlim(trf)
        p = plot._brain.BinTable(trf, fmin=0, fmax=vlim, tstep=tstep,
                                 summary='extrema', show=False, **surfer_kwargs)
        cbar = p.plot_colorbar(orientation='vertical', show=False)
        subsec.add_figure(caption % name, (p, cbar))
        cbar.close()
