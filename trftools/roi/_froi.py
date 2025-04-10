from eelbrain import NDVar, SourceSpace, complete_source_space, concatenate, morph_source_space, set_parc, xhemi


def symmetric_mask(
        mask: NDVar,
) -> NDVar:
    """Make a surface source space mask symmetric.

    Parameters
    ----------
    mask
        Binary mask on FSAverage.

    Examples
    --------
    Symmetric functional ROI from a test result::

        froi_source = result.p <= 0.05
        froi_symmetric = symmetric_mask(froi_source)
    """
    source = mask.get_dim('souce')
    assert isinstance(source, SourceSpace)
    assert source.subject == 'fsaverage'
    fsa_vertices = source.vertices
    mask_ = complete_source_space(mask)
    mask_ = set_parc(mask_, 'aparc')
    # morph both hemispheres to the left hemisphere
    mask_from_lh, mask_from_rh = xhemi(mask_)
    mask_lh = mask_from_lh + mask_from_rh
    # morph the new ROI to the right hemisphere
    mask_rh = morph_source_space(mask_lh, vertices_to=[[], mask_lh.source.vertices[0]], xhemi=True)
    # combine the two hemispheres
    mask_sym = concatenate([mask_lh, mask_rh], 'source')
    # morph the result back to the source brain
    mask_ = morph_source_space(mask_sym, source.subject, fsa_vertices)
    mask_ = set_parc(mask_, mask.source.parc)
    # convert to boolean mask (morphing involves interpolation, so the output is in floats)
    return round(mask_).astype(bool)
