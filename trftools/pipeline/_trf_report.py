from math import ceil

from eelbrain import plot, fmtxt
from eelbrain._stats.testnd import MultiEffectNDTest
from eelbrain._stats.spm import LMGroup
from eelbrain.fmtxt import Figure, Section, FMText


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
