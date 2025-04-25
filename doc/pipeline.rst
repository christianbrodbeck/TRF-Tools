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

.. seealso::

    - For TRF-Tools installation instructions see the `GitHub README <https://github.com/christianbrodbeck/TRF-Tools>`_.
    - For a complete example of a TRF-Experiment with data, see the `pipeline section of the Alice repository <https://github.com/Eelbrain/Alice/tree/main/pipeline>`_.

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

.. Warning::
   When you change the contents of a predictor file, this will **not** be automatically detected.
   In order to remove cached results that incorporated the outdated predictor, use :meth:`TRFExperiment.invalidate`.

.. _pipeline-models:

^^^^^^
Models
^^^^^^

When referring to mTRF models, models are sets of terms, each term specifying one predictor variable. For information on how to specify terms for different predictors see the :class:`FilePredictor` documentation. Models can be constructed by combine terms with ``+``, for example:

 - ``x="gammatone-1"`` is a model with a single predictor
 - ``x="gammatone-1 + gammatone-on-1"`` is a model with two predictor

To shorten long models specifications, named sub-models can be specified in :attr:`TRFExperiment.models`. For example, with::

    models = {
        "auditory-gammatone": "gammatone-8 + gammatone-on-8",
    }

The combined auditory model can then be invoked with ``auditory``. For example, the effect of acoustic onsets in the combined auditory model could be tested with ``x="auditory @ gammatone-on-8"``, which would internally expand to ``x="gammatone-8 + gammatone-on-8 @ gammatone-on-8"``.

Use :meth:`TRFExperiment.show_model_terms` to list all terms in a model, e.g.::

    >>> alice.show_model_terms("auditory-gammatone")
    #   term
    ------------------
    0   gammatone-8
    1   gammatone-on-8


Stimuli
-------

In order to load the correct predictors for a model term, the pipeline also needs to know what stimulus was presented in each trial.
For this, use the :attr:`TRFExperiment.stim_var` attribute to determine which event column is used as stimulus name.
The default is ``TRFExperiment.stim_var = 'stimulus'``.
Thus, given the following events::

    #    i_start   trigger   T        SOA      subject   stimulus
    -------------------------------------------------------------
    0    1863      1         3.726    57.618   S01       s1
    1    30672     5         61.344   60.898   S01       s2
    ...

Using the term 'gammatone' in a model would find predictor files based on the ``stimulus`` column: ``s1~gammatone.pickle``, ``s2~gammatone.pickle``, ...


Multiple stimuli per trial
--------------------------
To look up the stimlus in an event column other than the one specified in :attr:`TRFExperiment.stim_var`, simply specify the relevant column in the model term.
For example, assume a selective attention task in which two speakers talk simultaneousy.
The attended speaker ('fg') and the unattended speaker ('bg') can each be considered one stimulus.
In addition, the acoustic mixture of the two speakers may be considered a third stimulus ('mix').
Events may look like this::

    #    i_start   trigger   T        SOA      subject   fg   bg   mix
    ------------------------------------------------------------------
    0    1863      1         3.726    57.618   S01       s1   s3   s13
    1    30672     5         61.344   60.898   S01       s2   s4   s24
    ...


The default stimulus could be specified in ``TRFExperiment.stim_var = 'fg'``. Other stimuli (or "streams") could be specified in model terms with ``~``:

 - 'gammatone' would find predictors based on the default ``fg`` column: ``s1~gammatone.pickle``, ``s2~gammatone.pickle``, ...
 - 'bg~gammatone' would find predictors based on the ``bg`` column: ``s3~gammatone.pickle``, ``s4~gammatone.pickle``, ...
 - 'mix~gammatone' would find predictors based on the ``mix`` column: ``s13~gammatone.pickle``, ``s24~gammatone.pickle``, ...


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

Use :meth:`TRFExperiment.show_model_terms` to list the terms in the two models involved in a comparison::

    >>> alice.show_model_terms("auditory-gammatone @ gammatone-8")
    x1               x0
    -------------------------------
    gammatone-8
    gammatone-on-8   gammatone-on-8


^^^^^^^^^^^^^^^^^^^^^
Batch estimating TRFs
^^^^^^^^^^^^^^^^^^^^^

The pipeline computes and caches TRFs whenever they are requested through one of the methods for accessing results. However, it is often more expedient to estimate multiple TRF models before performing an analysis. This can be done by creating a list of TRF jobs, as in the `Alice jobs.py <https://github.com/Eelbrain/Alice/blob/pipeline/pipeline/jobs.py>`_ example. These TRFs can then be pre-computed by running the following command in a terminal:

.. code-block:: bash

    $ trf-tools-make-jobs jobs.py

When running this command, TRFs that have already been cached will be skipped automatically, so there is no need to remove previous jobs from `jobs.py`. For example, when adding new subjects to a dataset, this command can be used to compute all TRFs for the new subjects. The pipeline also performs a cache check for every TRF, so this is a convenient way to re-create all TRFs after, for example, changing a preprocessing parameter.
