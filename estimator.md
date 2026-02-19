# TRF Estimator Parameters

## NCRF estimator (`ncrf_args` + fixed)

I get this definition from `line 1066-1078`.

| Parameter   | Default / Source | Description |
|------------|-------------------|-------------|
| **mu**     | `'auto'`          | Regularization. Set by `inv` tag: no tag → `'auto'`; numeric tag (e.g. `ncrf-5000`) → `float(tag)/10000`; `ncrf-l2` → `cv.cv_mu('l2')`; `ncrf-l2mu` → `cv.cv_mu('l2/mu')`; `ncrf-cv2` → array from `np.logspace` around best mu. |
| **n_iter** | —                 | Only set for some tags: `ncrf-50it` → 50; `ncrf-no_champ` → 1. |
| **n_iterf**| —                 | Only for `ncrf-no_champ`: 1000. |
| **n_iterc**| —                 | Only for `ncrf-no_champ`: 0. |
| **normalize** | `True` (fixed)  | Passed to `fit_ncrf`, not in `ncrf_args`. |
| **in_place**  | `True` (fixed)  | Passed to `fit_ncrf`, not in `ncrf_args`. |

## Boosting estimator
I get this from  `partial(boosting, y, xs, tstart, tstop, 'inplace', delta, mindelta, error, basis, partitions=partitions, test=cv, selective_stopping=selective_stopping, partition_results=partition_results)`

| Parameter              | Default   | Description |
|------------------------|-----------|-------------|
| **tstart**             | 0         | TRF time window start (s). |
| **tstop**              | 0.5       | TRF time window stop (s). |
| **delta**              | 0.005     | Boosting delta. |
| **mindelta**           | None      | Boosting mindelta. |
| **error**              | `'l1'`    | Error function: `'l1'` or `'l2'`. |
| **basis**              | 0.050     | Response function basis window (s). |
| **partitions**         | inferred  | CV folds: positive = by trials; negative = concatenate then split. |
| **test** (cv)          | True      | Whether to run cross-validation. |
| **selective_stopping** | 0         | Stop boosting per predictor separately. |
| **partition_results**  | False     | Keep per test-partition TRFs and metrics. |
