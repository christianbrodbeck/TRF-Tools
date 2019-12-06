# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from eelbrain import load, NDVar, epoch_impulse_predictor, resample


class EventPredictor:
    """Generate an impulse at each event

    Parameters
    ----------
    value : scalar | str
        Name of a :class:`Var` or :class:`Factor` in the events :class:`Dataset`
        (or expression resulting in one).
    latency : scalar | str
        Latency of the impulse relative to the event in seconds (or expression
        retrieving it from the events dataset).
    """

    def __init__(self, value=1., latency=0.):
        self.value = value
        self.latency = latency

    def _generate(self, time, ds, code):
        assert code.stim is None
        return epoch_impulse_predictor((ds.n_cases, time), self.value, self.latency, code.code, ds)


class FilePredictor:
    """Predictor stored in file(s)"""
    def __init__(self, resample=None):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample

    def _load(self, path, tstep):
        x = load.unpickle(path)
        if isinstance(x, NDVar):
            if x.time.tstep == tstep:
                pass
            elif self.resample == 'bin':
                x = x.bin(tstep, label='start')
            elif self.resample == 'resample':
                srate = 1 / tstep
                int_srate = int(round(srate))
                srate = int_srate if abs(int_srate - srate) < .001 else srate
                x = resample(x, srate)
            elif self.resample is None:
                raise RuntimeError(f"{os.path.basename(path)} has tstep={x.time.tstep}, not {tstep}")
            else:
                raise RuntimeError(f"resample={self.resample!r}")
        else:
            xs = x
            for x in xs:
                if x.time.tstep == tstep:
                    break
            else:
                raise IOError(f"{os.path.basename(path)} does not contain tstep={tstep!r}")
        return x


class MakePredictor:
    """Predictor calls .make_predictor"""
