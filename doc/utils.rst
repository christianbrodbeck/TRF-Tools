*********
Utilities
*********

^^^^^^^^
TextGrid
^^^^^^^^
A class for working with Praat TextGrid forced alignment files.

.. currentmodule:: trftools
.. autosummary::
   :toctree: generated

    TextGrid

.. py:module:: dictionaries
.. currentmodule:: trftools.dictionaries

^^^^^^^^^^^^
Dictionaries
^^^^^^^^^^^^
The :mod:`dictionaries` module contains functions for retrieving and manipulating linguistic data, including SUBTLEX corpus statistics and pronunciation dictionaries.

.. autosummary::
   :toctree: generated

    read_cmupd
    read_dict
    combine_dicts
    split_apostrophe
    write_dict
    read_subtlex
    read_subtlex_pos
    fix_apostrophe_pronounciations


.. py:module:: roi
.. currentmodule:: trftools.roi

^^^
ROI
^^^
The :mod:`roi` module contains functions for working with anatomical ROIs.

.. autosummary::
   :toctree: generated

    mask_roi
    mne_label
    symmetric_mask
