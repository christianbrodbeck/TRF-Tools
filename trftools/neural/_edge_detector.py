# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from scipy.signal.windows import gaussian

from eelbrain import NDVar
import numpy as np
from scipy.signal import convolve2d


def apply_receptive_field(stimulus, rf, clip=True, name=None):
    """Temporal suppression

    Two ways of conceptualizing:

    - STRF with center-surround field, edge detector
    - Envelope input exerts suppression of responses in the future
    """
    if name is None:
        name = stimulus.name
    tdim = stimulus.get_dim('time')
    fdim = stimulus.get_dim('frequency')
    assert tdim.tstep == 0.001
    stim_data = stimulus.get_data(('frequency', 'time'))
    out = _apply_rf_array(stim_data, rf, clip)
    return NDVar(out, (fdim, tdim), name, stimulus.info)


def _apply_rf_array(stim, rf, clip):
    out = convolve2d(stim, rf, 'full')

    n_y_over = rf.shape[0] - 1
    y_lower = n_y_over // 2
    y_upper = y_lower - n_y_over
    if y_lower:
        out = out[y_lower: y_upper, :stim.shape[1]]
    else:
        out = out[:, :stim.shape[1]]

    if clip:
        out = out.clip(0, out=out)
    return out


#####################
# Edge detector model
#####################

def delay_neuron(tau):
    lag = np.arange(40)  # t - x
    return np.e**(-lag / tau) * lag / tau**2


def delay_rf(tau):
    return delay_neuron(tau)[None, :]


def saturate(x: NDVar, c: float = 10):
    x_out = 2 / (1 + np.e**(-x.x / c)) - 1
    return NDVar(x_out, x.dims, name=f'c={c}')


def edge_detector(
        gammatone: NDVar,
        c: float = 0,
        offset: bool = False,
        name: str = None,
):
    """Neural model for auditory edge-detection, as described by [1]_ and used in [2]_

    Parameters
    ----------
    gammatone
        Gammatone spectrogram.
    c
        Saturation parameter (see [1]_).
    offset
        Detect offsets (instead of onsets).
    name
        Name for the returned :class:`NDVar`.

    References
    ----------
    .. [1] Fishbach, A., Nelken, I., & Yeshurun, Y. (2001). Auditory Edge Detection: A Neural Model for Physiological and Psychoacoustical Responses to Amplitude Transients. Journal of Neurophysiology, 85(6), 2303–2323. https://doi.org/10.1152/jn.2001.85.6.2303
    .. [2] Brodbeck, C., Jiao, A., Hong, L. E., & Simon, J. Z. (2020). Neural speech restoration at the cocktail party: Auditory cortex recovers masked speech of both attended and ignored speakers. PLOS Biology, 18(10), e3000883. https://doi.org/10.1371/journal.pbio.3000883

    """
    taus = np.linspace(3, 5, 10)
    ws = np.diff(gaussian(11, 2))
    if offset:
        ws *= -1
    rfs = [delay_rf(tau) for tau in taus]
    xs_d = [apply_receptive_field(gammatone, rf) for rf in rfs]
    if c:
        xs_d = [saturate(x, c) for x in xs_d]
    return sum([w * x for w, x in zip(ws, xs_d)]).clip(0, name=name)
