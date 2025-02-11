# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
import re
from typing import List, Literal, Union

import mne
from nibabel.freesurfer import read_annot
import numpy

from eelbrain import NDVar, SourceSpace, set_parc
from eelbrain._types import PathArg


HEMIS = ('lh', 'rh')
ROIS = {
    'STG': ['transversetemporal', 'superiortemporal'],
    'STS': ['bankssts'],
    'MTG': ['middletemporal'],
    'ITG': ['inferiortemporal'],
    'FF': ['fusiform'],
    'IFG': ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
}
HEMI_ROIS = {
    key: {hemi: [f'{label}-{hemi}' for label in labels] for hemi in HEMIS}
    for key, labels in ROIS.items()
}
for key in ROIS:
    HEMI_ROIS[key]['both'] = HEMI_ROIS[key]['lh'] + HEMI_ROIS[key]['rh']


def roi_to_aparc(roi: str) -> List[str]:
    "Convert roi spec to list of aparc labels"
    if roi in ROIS:
        return ROIS[roi]
    elif isinstance(roi, str):
        return [roi]
    else:
        out = list(roi)
        assert all(isinstance(label, str) for label in out)
        return out


def mne_label(
        key: str,
        subject: str = 'fsaverage',
        subjects_dir: PathArg = None,
        hemi: Literal['lh', 'rh', 'both'] = 'both',
) -> Union[mne.Label, mne.BiHemiLabel]:
    """MNE label object corresponding to a ROI

    Parameters
    ----------
    key
        Label definition (see Notes to :func:`mask_roi`).
    subject
        Name of the MRI-subject (FreeSurfer ``SUBJECT``).
    subjects_dir
        MRI directory (FreeSurfer ``SUBJECTS_DIR``).
    hemi
        Hemisphere.
    """
    if hemi == 'both':
        return mne_label(key, subject, subjects_dir, 'lh') + mne_label(key, subject, subjects_dir, 'rh')
    hemi_vertices = vertices(key, subject, subjects_dir, hemi)
    return mne.Label(hemi_vertices, hemi=hemi, name=f'{key}-{hemi}', subject=subject)


def mask_roi(
        key: str,
        src: Union[SourceSpace, Path, str],
) -> (NDVar, NDVar):  # lh, rh
    """Create a boolean ROI based on anatomical labels

    Parameters
    ----------
    key
        Label definition (see Notes).
    src
        Source space to work with, or MRI-directory from which to load fsaverage.

    Returns
    -------
    ``[lh, rh]`` Boolean masks for both hemisphere (``True`` inside the label).

    Notes
    -----
    Labels are defined based on a label name, and optional instructions for
    splitting the label.

    The label name can be one of the pre-defined collections defined in
    ``ROIS`` at the top of this script, e.g. ``STG``.
    It can also be the name of an aparc label, e.g. ``middletemporal``.

    Splitting instructions is based on number of parts, followed by the parts
    to use (with zero-based index, from posterior to anterior).
    E.g. ``STG301``: split STG into 3 parts, and use the posterior 2/3rd
    (parts 0 and 1).

    Multiple such definitions can be combined with ``+``, e.g. ``STG+STS``.
    """
    if isinstance(src, str) or isinstance(src, Path):
        src = SourceSpace.from_file(src, 'fsaverage', 'ico-4', 'aparc')
    elif src.parc.name != 'aparc':
        src = set_parc(src, 'aparc')
    hemi_labels = []
    for hemi in HEMIS:
        hemi_vertices = vertices(key, src.subject, src.subjects_dir, hemi)
        index = numpy.in1d(src.vertices[hemi == 'rh'], hemi_vertices)
        if hemi == 'lh':
            index = numpy.concatenate([index, numpy.zeros(src.rh_n, bool)])
        else:
            index = numpy.concatenate([numpy.zeros(src.lh_n, bool), index])
        hemi_labels.append(NDVar(index, src, name=f'{key}-{hemi}'))
    return hemi_labels


def vertices(
        key: str,
        subject: str = 'fsaverage',
        subjects_dir: PathArg = None,
        hemi: Literal['lh', 'rh'] = 'both',
) -> numpy.ndarray:
    """Vertices corresponding to a ROI"""
    assert subjects_dir is not None, "subjects_dir must be specified"
    path = Path(subjects_dir) / subject / 'label' / f'{hemi}.aparc.annot'
    labels, ctab, names = read_annot(path)

    # prepare RE
    pattern = re.compile(r"([a-zA-Z]+)(\d*)")
    key_parts = key.split('+')
    roi_parts = []
    for key_part in key_parts:
        m = pattern.match(key_part)
        if not m:
            raise ValueError(f"{key=}")
        base, split = m.groups()
        base_ids = [names.index(base_key.encode()) for base_key in roi_to_aparc(base)]
        part_vertices = numpy.flatnonzero(numpy.in1d(labels, base_ids))
        # label sub-section
        if split:
            n, *keep = [int(c) for c in split]
            if not keep:
                raise ValueError(f"{key=}")
            label = mne.Label(part_vertices, hemi=hemi, name=base, subject=subject)
            label_parts = label.split(n, subject, subjects_dir)
            part_vertices = numpy.concatenate([label_parts[i].vertices for i in keep])
            part_vertices.sort()
        roi_parts.append(part_vertices)
    all_vertices = numpy.concatenate(roi_parts)
    all_vertices.sort()
    return all_vertices


