import re
from pathlib import Path

import eelbrain
from eelbrain import concatenate, save


DATA_ROOT = Path("/Users/yanyuwoo/Data/Appleseed_BIDS_20251216")
WAV_DIR = Path("/Users/yanyuwoo/Data/stimuli")
PREDICTOR_DIR = DATA_ROOT / "derivatives" / "predictors"
OUTPUT_PATH = PREDICTOR_DIR / "Appleseed~acoustic_envelop.pickle"

SAMPLINGRATE = 100
LOW_FREQUENCY = 0.5
HIGH_FREQUENCY = 20


def _wav_sort_key(path: Path) -> tuple[float, str]:
    match = re.search(r"(\d+)", path.stem)
    number = int(match.group(1)) if match else float("inf")
    return number, path.name


def main():
    if not WAV_DIR.exists():
        raise FileNotFoundError(f"Stimulus directory not found: {WAV_DIR}")

    wav_paths = sorted((path for path in WAV_DIR.iterdir() if path.suffix.lower() == ".wav"), key=_wav_sort_key)
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found in: {WAV_DIR}")

    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

    wavs = [eelbrain.load.wav(path) for path in wav_paths]
    wav = concatenate(wavs)
    envelope = wav.envelope()
    envelope = eelbrain.filter_data(envelope, LOW_FREQUENCY, HIGH_FREQUENCY, pad="reflect")
    envelope = eelbrain.resample(envelope, SAMPLINGRATE)
    envelope.name = "acoustic_envelop"

    save.pickle(envelope, OUTPUT_PATH)
    print(f"Wrote predictor: {OUTPUT_PATH}")
    print(f"Used {len(wav_paths)} wav files from {WAV_DIR}")
    print(f"Predictor tstep: {envelope.time.tstep}")
    print(f"Predictor nsamples: {envelope.time.nsamples}")


if __name__ == "__main__":
    main()
