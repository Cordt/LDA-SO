__author__ = 'Cordt'

from scipy import stats
import numpy as np


def js_distance(p, q):
    return np.sqrt(js_divergence(p, q))


def js_divergence(p, q):
    _p = p / np.linalg.norm(p, ord=1)
    _q = q / np.linalg.norm(q, ord=1)
    _m = 0.5 * (_p + _q)
    return 0.5 * (stats.entropy(_p, _m) + stats.entropy(_q, _m))
