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

To get started, set up your experiment as for the `MNE-Experiment pipeline <https://eelbrain.readthedocs.io/en/stable/experiment.html>`_, but instead of :class:`eelbrain.pipeline.MneExperiment`, use :class:`TRFExperiment` as baseclass for your experiment. The :class:`TRFExperiment` uses :class:`~eelbrain.pipeline.MneExperiment` mechanisms to preprocess data up to the epoch stage.

For an existing experiment, the base class can simply be replaced, because :class:`TRFExperiment` retains all functionality of :class:`~eelbrain.pipeline.MneExperiment`.

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

Assuming a stimulus called ``story``, this would match the following predictor files:

 - ``{root}/predictors/story~gammatone-1.pickle``: :class:`NDVar` UTS predictor, which can be invoked with model term ``gammatone-1`` (see :ref:`pipeline-models`)
 - ``{root}/predictors/story~gammatone-8.pickle``: as above, but invoked with model term ``gammatone-8``
 - ``{root}/predictors/story~word.pickle``: a :class:`Dataset` representing one or multiple NUTS predictors (through different columns in the dataset). The specific model term would include a column name, for example, a model term ``word-surprisal`` would use the values of the ``"surprisal"`` column in the dataset (see :class:`FilePredictor`).


.. _pipeline-models:

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

When referring to model tests, this usually means comparing two different mTRF models. Basic comparisons can be constructed with ``>``, ``<`` (one-tailed) and ``=`` (two-tailed):

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


^^^^^^^^^^^^^^^^^^^^^
Batch estimating TRFs
^^^^^^^^^^^^^^^^^^^^^

The pipeline computes and caches TRFs whenever they are requested through one of the methods for accessing results. However, it is often more expedient to estimate multiple TRF models before performing an analysis. This can be done by creating a list of TRF jobs, as in the `Alice jobs.py <https://github.com/Eelbrain/Alice/blob/pipeline/pipeline/jobs.py>`_ example. These TRFs can then be pre-computed by running the following command in a terminal:

.. code-block:: bash

    $ trf-tools-make-jobs jobs.py

When running this command, TRFs that have already been cached will be skipped automatically, so there is no need to remove previous jobs from `jobs.py`. For example, when adding new subjects to a dataset, this command can be used to compute all TRFs for the new subjects. The pipeline also performs a cache check for every TRF, so this is a convenient way to re-create all TRFs after, for example, changing a preprocessing parameter.
