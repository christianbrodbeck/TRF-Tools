# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from trftools import notebooks  # register backend
import joblib


def test_joblib():
    memory = joblib.Memory('.', 'memory')

    @memory.cache
    def get_double(x):
        return x * 2

    assert get_double(5) == 10
    assert get_double(6) == 12
    assert get_double(5) == 10
    assert get_double(6) == 12

    # make sure the result cached, not recomputed
    @memory.cache
    def get_list(x):
        return [x]

    x = get_list(9)
    assert x == [9]
    x.append(1)
    x2 = get_list(9)
    assert x2 == [9, 1]
