"""Visualize a cached TRF/NCRF result pickle.

Examples
--------
python trftools/pipeline/estimator/demo/visualize_result.py \
    "/Users/yanyuwoo/Downloads/sub-R2349_split-01_meg nobl 0-500 100Hz acoustic_envelop boosting h50 l1 seg cv.pickle"

The script accepts Eelbrain/TRF-Tools result pickles and Dataset pickles. It is
meant as a quick visual reasonableness check: inspect the TRF time course,
sensor topographies, and volume-source glass-brain snapshots when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib


def _log(message: str) -> None:
    print(message, flush=True)


def _load_pickle(path: Path):
    try:
        from eelbrain import load
    except ImportError as error:
        raise RuntimeError(
            "Eelbrain is required to load TRF result pickles. Activate the TRF "
            "environment first, for example `mamba activate trf`."
        ) from error

    _avoid_local_ncrf_shadow()
    try:
        return load.unpickle(path)
    except ModuleNotFoundError as error:
        if error.name == "ncrf":
            raise RuntimeError(
                "This pickle references the `ncrf` package, but it is not "
                "installed in the active Python environment. Run this script in "
                "the same environment used to fit the result, or install ncrf "
                "there, then try again."
            ) from error
        raise


def _avoid_local_ncrf_shadow() -> None:
    """Do not let estimator/ncrf.py shadow the external ncrf package.

    Direct script execution adds this file's directory to sys.path. The result
    pickle may need to import the external package named ``ncrf``; without this
    cleanup Python can import the adjacent estimator config module instead.
    """
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [path for path in sys.path if Path(path or ".").resolve().as_posix() != script_dir]


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text).strip("_") or "trf"


def _dimnames(ndvar) -> tuple[str, ...]:
    return tuple(dim.name for dim in ndvar.dims)


def _has_dim(ndvar, dim: str) -> bool:
    return dim in _dimnames(ndvar)


def _as_sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _iter_ndvars(obj, preferred: str) -> list:
    """Find plausible TRF NDVars in a loaded result object."""
    from eelbrain import Dataset, NDVar

    found = []

    if isinstance(obj, NDVar):
        return [(obj.name or "ndvar", obj)]

    if isinstance(obj, Dataset):
        keys = []
        keys.extend(obj.info.get("xs", ()))
        keys.extend(k for k in obj.keys() if k not in keys)
        for key in keys:
            if key in obj and isinstance(obj[key], NDVar):
                y = obj[key]
                if _has_dim(y, "case") and len(y.get_dim("case")) == 1:
                    y = y.sub(case=0)
                if _has_dim(y, "time"):
                    found.append((key, y))
        return found

    for attr in (preferred, "h_scaled", "h", "h_source"):
        if not hasattr(obj, attr):
            continue
        for i, y in enumerate(_as_sequence(getattr(obj, attr))):
            if isinstance(y, NDVar) and _has_dim(y, "time"):
                label = f"{attr}_{y.name or 'trf'}"
                if i:
                    label = f"{label}-{i + 1}"
                found.append((label, y))

    # Keep order but avoid duplicate objects from h/h_scaled aliases.
    out = []
    seen = set()
    for label, y in found:
        ident = (id(y.x), tuple(dim.name for dim in y.dims), y.shape)
        if ident not in seen:
            out.append((label, y))
            seen.add(ident)
    return out


def _summarize_result(obj, ndvars: Sequence[tuple[str, object]]) -> None:
    print(f"Loaded: {type(obj).__module__}.{type(obj).__name__}")
    for attr in ("x", "mu", "partitions", "n_samples", "tstep"):
        if hasattr(obj, attr):
            print(f"{attr}: {getattr(obj, attr)!r}")

    for attr in ("r", "r_l1", "residual", "proportion_explained"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            dims = getattr(value, "dims", None)
            shape = getattr(value, "shape", None)
            print(f"{attr}: {type(value).__name__}, dims={dims}, shape={shape}")

    print("TRF NDVars:")
    for label, y in ndvars:
        vmin = float(y.min())
        vmax = float(y.max())
        print(
            f"  {label}: dims={_dimnames(y)}, shape={y.shape}, "
            f"min={vmin:.6g}, max={vmax:.6g}"
        )
        if vmin == 0 and vmax == 0:
            print("    WARNING: all TRF values are zero; figures will be blank/flat.")


def _times_from_ndvar(y, requested: Sequence[float] | None, max_times: int) -> list[float]:
    if requested:
        return list(requested)
    if not _has_dim(y, "time"):
        return []
    tmin = float(y.time.tmin)
    tstop = float(y.time.tstop)
    if max_times <= 1:
        return [(tmin + tstop) / 2]
    step = (tstop - tmin) / (max_times + 1)
    return [tmin + step * (i + 1) for i in range(max_times)]


def _prepare_for_spatial_plot(y):
    """Collapse vector/direction dimensions while preserving time and sensors/source."""
    dims = _dimnames(y)
    if "space" in dims:
        return y.norm("space")
    if "time" in dims and ("sensor" in dims or "source" in dims) and len(dims) > 2:
        collapse = [dim for dim in dims if dim not in ("time", "sensor", "source")]
        for dim in collapse:
            y = y.rms(dim)
    return y


def _save_plot(plot_obj, path: Path) -> None:
    plot_obj.save(str(path))
    plot_obj.close()
    _log(f"saved {path}")


def _plot_ndvar(
    label: str,
    y,
    out_dir: Path,
    times: Sequence[float] | None,
    max_times: int,
    surface_brain: bool,
) -> None:
    from eelbrain import plot

    y = _prepare_for_spatial_plot(y)
    stem = _safe_name(label)

    butterfly = plot.Butterfly(y, show=False, title=f"{label} TRF")
    _save_plot(butterfly, out_dir / f"{stem}_butterfly.png")

    if _has_dim(y, "sensor"):
        topo = plot.TopoArray(
            y,
            t=_times_from_ndvar(y, times, max_times),
            title=f"{label} topographies",
            show=False,
        )
        _save_plot(topo, out_dir / f"{stem}_topographies.png")
    elif _has_dim(y, "source"):
        source = y.get_dim("source")
        is_volume = source.__class__.__name__ == "VolumeSourceSpace"
        if is_volume:
            brain = plot.GlassBrain.butterfly(
                y,
                name=f"{label} volume source TRF",
                colorbar=True,
                display_mode="ortho",
            )
            _save_plot(brain, out_dir / f"{stem}_glassbrain_butterfly.png")
            for time in _times_from_ndvar(y, times, max_times):
                yt = y.sub(time=time)
                brain = plot.GlassBrain(yt, show=False, colorbar=True, display_mode="ortho")
                _save_plot(brain, out_dir / f"{stem}_glassbrain_{time * 1000:.0f}ms.png")
        elif not surface_brain:
            _log(
                "skipped surface brain plot because it can open GUI windows; "
                "use --surface-brain if you want the interactive surface view"
            )
        else:
            brain = plot.brain.butterfly(y, name=f"{label} source TRF")
            _save_plot(brain, out_dir / f"{stem}_source_butterfly.png")


def _parse_times(values: Iterable[str] | None) -> list[float] | None:
    if not values:
        return None
    return [float(value) for value in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle", type=Path, help="Path to an Eelbrain/TRF result pickle.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PNG files. Defaults to <pickle-name>_figures beside the pickle.",
    )
    parser.add_argument(
        "--preferred",
        default="h_scaled",
        help="Preferred result attribute to visualize first, usually h_scaled or h.",
    )
    parser.add_argument(
        "--time",
        nargs="+",
        default=None,
        help="Snapshot times in seconds. Default: evenly spaced times across the TRF window.",
    )
    parser.add_argument("--max-times", type=int, default=5, help="Number of default snapshot times.")
    parser.add_argument("--show", action="store_true", help="Use the interactive matplotlib backend.")
    parser.add_argument(
        "--surface-brain",
        action="store_true",
        help="Also create interactive surface brain plots for surface-source results. This can open GUI windows.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Print pickle contents without making plots.")
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")
        import eelbrain

        eelbrain.configure(frame=False, autorun=False, show=False)

    path = args.pickle.expanduser().resolve()
    out_dir = args.out or path.with_suffix("").with_name(path.stem + "_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Input pickle: {path}")
    _log(f"Output directory: {out_dir}")
    _log("Loading pickle...")

    try:
        obj = _load_pickle(path)
        _log("Pickle loaded.")
        ndvars = _iter_ndvars(obj, args.preferred)
        if not ndvars:
            raise RuntimeError(
                "No time-resolved NDVar was found. Expected an NDVar, an Eelbrain "
                "Dataset with TRF columns, or a result object with h/h_scaled/h_source."
            )

        _summarize_result(obj, ndvars)
        if args.inspect_only:
            _log("Inspect-only mode: no figures were created.")
            return 0
        times = _parse_times(args.time)
        for label, y in ndvars:
            _log(f"Plotting {label}...")
            _plot_ndvar(label, y, out_dir, times, args.max_times, args.surface_brain)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    _log(f"Done. Figure directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
