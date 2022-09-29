from mcerp import correlate
from mcerp import N
from mcerp import stats

import numpy as np
from scipy import stats as stats

target_corr = [
    [1, 0.2, 0.2, 0.2, 0.2, 0.8],
    [0.2, 1, 0.2, 0.2, 0.2, 0.6],
    [0.2, 0.2, 1, 0.2, 0.2, 0.4],
    [0.2, 0.2, 0.2, 1, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 1, 0],
    [0.8, 0.6, 0.4, 0.2, 0, 1],
]


def generate_data():
    v1 = N(0, 1)
    v2 = N(0, 1)
    v3 = N(0, 1)
    v4 = N(0, 1)
    v5 = N(0, 1)
    v6 = N(0, 1)
    c_target = np.array(target_corr)
    correlate([v1, v2, v3, v4, v5, v6], c_target)
    rv1 = v1._mcpts
    rv2 = v2._mcpts
    rv3 = v3._mcpts
    rv4 = v4._mcpts
    rv5 = v5._mcpts
    rv6 = v6._mcpts
    data = np.vstack((rv1, rv2, rv3, rv4, rv5, rv6)).T
    labels = data[:, -1]  # for last column
    inp_data = data[:, :-1]  # for all but last column
    return inp_data, labels
