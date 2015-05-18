__author__ = 'Cordt'

from scipy import stats
import numpy as np


def jsdistance(p, q):
    return np.sqrt(jsdivergence(p, q))


def jsdivergence(p, q):
    _p = p / np.linalg.norm(p, ord=1)
    _q = q / np.linalg.norm(q, ord=1)
    _m = 0.5 * (_p + _q)
    return 0.5 * (stats.entropy(_p, _m) + stats.entropy(_q, _m))
