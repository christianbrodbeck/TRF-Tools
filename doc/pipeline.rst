.. py:module:: pipeline
.. currentmodule:: trftools.pipeline

********
Pipeline
********
The :mod:`pipeline` module provides an extension of the Eelbrain `MNE-Experiment pipeline <https://eelbrain.readthedocs.io/en/stable/experiment.html>`_ for mTRF analysis.

^^^^^^^^^^^^^^^^
Module Reference
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

    TRFExperiment
    FilePredictor
    EventPredictor

^^^^^
Setup
^^^^^

For a complete example, see the `Alice repository <https://github.com/Eelbrain/Alice/tree/pipeline/pipeline>`_.

To get started, set up your experiment as for the `MNE-Experiment pipeline <https://eelbrain.readthedocs.io/en/stable/experiment.html>`_, but instead of :class:`eelbrain.MneExperiment`, use :class:`TRFExperiment` as baseclass for your experiment. For an existing experiment, the base class can simply be replaced, because :class:`TRFExperiment` retains all functionality of :class:`eelbrain.MneExperiment`.


^^^^^^^^^^
Predictors
^^^^^^^^^^

Predictors are added to the experiment as files in the ``{root}/predictors`` directory. Filenames should follow this pattern: ``{stimulus}~{key}[-{variant}].pickle``.

 - ``stimulus`` referes to an arbitrary name for the stimulus represented by this file (see :attr:`TRFExperiment.stim_var`).
 - ``key`` is the key used for defining this predictor in :attr:`TRFExperiment.predictors`.
 - ``variant`` is an optional description that allows several predictors using the same entry in :attr:`TRFExperiment.predictors`. That allows, for example, only defining a single ``gammatone`` predictor in :attr:`TRFExperiment.predictors` for different variations of the spectrogram (``gammatone-1``, ``gammatone-8``, ``gammatone-on-1``, etc.).

Predictors are then added to the pipeline in :class:`TRFExperiment.predictors`. This is a dictionary in which the key is the ``key`` mentioned above, and the value is typically a :class:`FilePredictor` object. For information on how to handle different kinds of predictors see the :class:`FilePredictor` documentation. For example::

    predictors = {
        'gammatone': FilePredictor(resample='bin'),
        'word': FilePredictor(columns=True),
    }


^^^^^^
Models
^^^^^^

When referring to mTRF models, models are sets of terms, each term specifying one predictor variable. For information on how to specify terms for different predictors see the :class:`FilePredictor` documentation. Models can be constructed by combine terms with ``+``, for example:

 - ``x="gammatone-1"`` is a model with a single predictor
 - ``x="gammatone-1 + gammatone-on-1"`` is a model with two predictor

To shorten long models specifications, named sub-models can be specified in :attr:`TRFExperiment.models`. For example, with::

    models = {
        "auditory": "gammatone-8 + gammatone-on-8",
    }

The combined auditory model can then be invoked with ``auditory``. For example, the effect of acoustic onsets in the combined auditory model could be tested with ``x="auditory @ gammatone-on-8"``, which would internally expand to ``x="gammatone-8 + gammatone-on-8 @ gammatone-on-8"``.


.. _trf-experiment-comparisons:

^^^^^^^^^^^
Comparisons
^^^^^^^^^^^

When referring to model tests, this usually means comparing two different mTRF models. Basic comparisons can be constructed with ``>``/``<`` (one-tailed) and ``=`` (two-tailed):

 - ``x="gammatone-1 + gammatone-on-1 > gammatone-1"`` tests whether predictive power improves when adding the ``gammatone-on-1`` predictor to a model already containing the ``gammatone-1`` predictor.
 - ``x="gammatone-1 = gammatone-on-1"`` tests whether the predictive power of ``gammatone-1`` or that of ``gammatone-on-1`` is higher.

To simplify common tests with large models, the following shortcuts exist:

.. list-table::
   :header-rows: 1
   :widths: 10 20 25 45

   * - Shortcut
     - Example
     - Expansion
     - Description
   * - ``@``
     - ``a + b + c @ a``
     - ``a + b + c > b + c``
     - Unique contribution of ``a`` to the left-hand-side model
   * - ``+@``
     - ``b + c +@ a``
     - ``a + b + c > b + c``
     - Effect of adding ``a`` to the left-hand-side model
