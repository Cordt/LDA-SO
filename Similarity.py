__author__ = 'Cordt'

from scipy.stats import entropy
from numpy.linalg import norm


def jsd(p, q):
    _p = p / norm(p, ord=1)
    _q = q / norm(q, ord=1)
    _m = 0.5 * (_p + _q)
    return 0.5 * (entropy(_p, _m) + entropy(_q, _m))