from pathlib import Path
from eelbrain import load, combine
from eelbrain.pipeline import RawFilter, RawICA, RawApplyICA, RawMaxwell, RawSource, PrimaryEpoch, LabelVar

from trftools.pipeline import TRFExperiment, FilePredictor
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


def _log(msg: str):
    print(f"[demo] {msg}")


DIR = Path(__file__).parent
STIMULI_LENGTHS = load.tsv(DIR / "appleseed_stimuli.txt", types="fv")
STIMULI_PILOTS = STIMULI_LENGTHS.sub("stimulus != '11'").repeat(2)
STIMULI_REAL = STIMULI_LENGTHS.sub("stimulus != '11b'").repeat(2)


def _ensure_demo_predictor(e):
    """Check that the real demo predictor exists."""
    pred_dir = Path(e.get("predictor-dir"))
    target = pred_dir / "Appleseed~acoustic_envelop.pickle"
    if not target.exists():
        raise FileNotFoundError(f"Required predictor file not found: {target}")
    _log(f"Using predictor {target.name}.")


# Path for BIDS (Brain Imaging Data Structure) data
# example:
# sub-R2349/ (subject)
#   ├── meg/ (meg raw data)
#   ├── anat/ (mri data)
# derivatives/
#   ├── predictors/ (predictor files)
#   ├── ica/ (noise removal results)
#   ├── freesurfer/ (map MEG signals from sensors → brain surfaces)
#   ├── trans/ (coordinate alignment)
#   ├── eelbrain/ (Store intermediate results used by the TRF pipeline)

DATA_ROOT = "/Users/yanyuwoo/Data/Appleseed_BIDS_20251216"


class AppleSeed(TRFExperiment):
    data_dir = "meg"
    subject_re = r"sub-[A-Za-z0-9]+"  # BIDS subjects, e.g. sub-01 or sub-R2677
    sessions = ["Appleseed"]  # must match BIDS task name in filenames

    raw = {
        "raw": RawSource(connectivity="auto"),
        "tsss": RawMaxwell("raw", st_duration=10.0, ignore_ref=True, st_correlation=0.9, st_only=True),
        "1-40": RawFilter("tsss", 1, 40),
        "ica": RawICA("1-40", "Appleseed", n_components=0.99),
        "tsss-ica": RawApplyICA("tsss", "ica", cache=False),
        "0.5-20": RawFilter("tsss-ica", 0.5, 20, cache=False),
    }
    defaults = {"epoch": "Appleseed", "raw": "0.5-20"}

    # At least one epoch required for load_trf. Task name must match BIDS (e.g. task-Appleseed in filenames).
    epochs = {
        "Appleseed": PrimaryEpoch("Appleseed", "event == 'onset'", tmin=0, tmax="length", samplingrate=100),
        # Minimal covariance epoch for source-space/NCRF demos.
        "cov": PrimaryEpoch("Appleseed", None, tmin=-0.100, tmax=0.0, samplingrate=100),
    }

    # Predictor must exist as derivatives/predictors/{stimulus}~acoustic_envelop.pickle
    predictors = {
        "acoustic_envelop": FilePredictor(),
    }

    models = {
        "acoustic_envelop": "acoustic_envelop",
    }

    # Pipeline needs a "stimulus" column to load predictors. Eelbrain derives
    # events from the raw stimulus channel here, so use trigger codes rather
    # than relying on columns from the BIDS events.tsv.
    variables = {
        "event": LabelVar("trigger", {162: "onset", 167: "offset"}),
        "stimulus": LabelVar("trigger", {(162, 167): "Appleseed"}),
    }
    tests = {}

    # Estimators registry
    estimators = {
        # partitions required when n_cases not in 3..10 (e.g. 22 trials)
        "boosting": BoostingEstimator(basis=0.050, partitions=5),
        "boosting-l2": BoostingEstimator(error="l2", partitions=5),
        "ncrf": NCRFEstimator(mu=0.001, n_iter=100),
        "ncrf-fast": NCRFEstimator(mu=0.01, n_iter=10),
    }

    def fix_events(self, ds):
        if ds.info.get("subject") == "R2676":
            return combine([ds[:10], ds[11:]])
        return ds

    def label_events(self, ds):
        if ds.info.get("subject") in ("R2650", "R2652"):
            lengths = STIMULI_PILOTS["length"]
        else:
            lengths = STIMULI_REAL["length"]
        if len(lengths) != ds.n_cases:
            raise RuntimeError(f"Expected {len(lengths)} event lengths for {ds.info.get('subject')}, got {ds.n_cases} events")
        ds["length"] = lengths
        return ds

# Initialize the demo experiment
def _init_demo_experiment() -> AppleSeed:
    e = AppleSeed(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if not subjects:
        raise RuntimeError(f"No non-emptyroom subjects found under {DATA_ROOT}")
    subject = subjects[0]
    e.set(subject=subject)
    return e

# Run the demo
def _run_demo(estimator: str, **load_trf_kwargs):
    e = _init_demo_experiment()
    _log(f"Initialized experiment with DATA_ROOT={DATA_ROOT}")
    _log(f"Subject={e.get('subject')}, Epoch={e.get('epoch')}, Estimator={estimator!r}, Options={load_trf_kwargs!r}")
    _ensure_demo_predictor(e)
    if load_trf_kwargs.get("data") == "meg":
        _log("Using sensor-space MEG data.")
    _log(f"Running TRF with estimator={estimator!r} (make=True)...")
    trf = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        estimator=estimator,
        make=True,
        **load_trf_kwargs,
    )
    _log(f"Done: {trf}")
    return trf


def run_boosting_demo():
    """Recommended sensor-space boosting demo using the estimator registry."""
    return _run_demo("boosting", data="meg")


def run_ncrf_demo():
    """Recommended NCRF demo with estimator-managed sensor-space semantics."""
    return _run_demo("ncrf")


if __name__ == "__main__":
    run_boosting_demo()
    run_ncrf_demo()
