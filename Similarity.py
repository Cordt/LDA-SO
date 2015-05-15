__author__ = 'Cordt'

from scipy import stats
import scipy as scp
import numpy as np
import itertools
import math


def jsd(document, query, num_topics=100):
    # document = document mit (topic, topic weight) tupeln
    # query = query mit (topic, topic weight) tupeln
    # num_topics ist die Anzahl aller Topics
    return_sum = 0
    for current_topic in range(0, num_topics):
        document_topic_weight = 0
        query_topic_weight = 0
        subsum = 0
        for (topic, topic_weight) in document:
            if topic == current_topic:
                document_topic_weight = topic_weight
                #print "doc_weight: " + str(document_topic_weight)
        for (topic, topic_weight) in query:
            if topic == current_topic:
                query_topic_weight = topic_weight
                #print "query_weight: " + str(query_topic_weight)
        if document_topic_weight != 0:
            subsum = document_topic_weight * math.log( (document_topic_weight) / (0.5*(document_topic_weight+query_topic_weight)))
            #print "Run: " + str(current_topic) + " 1st subsum: " + str(subsum)
        if query_topic_weight != 0:
            subsum += query_topic_weight * math.log( (query_topic_weight) / (0.5*(document_topic_weight+query_topic_weight)))
            #print "Run: " + str(current_topic) + " 2nd subsum: " + str(subsum)
        #print "Run: " + str(current_topic) + " subsum = " + str(subsum)
        return_sum += subsum
    return return_sum


def jsdistance(p, q):
    np.sqrt(jsdivergence(p, q))


def jsdivergence(p, q):
    _p = p / np.linalg.norm(p, ord=1)
    _q = q / np.linalg.norm(q, ord=1)
    _m = 0.5 * (_p + _q)
    return 0.5 * (stats.entropy(_p, _m) + stats.entropy(_q, _m))


def jensen_shannon_divergence(freq, weights=None, unit='bit'):
    """
    Calculates the Jensen-Shannon Divergence (Djs) of two or more frequencies.
    The weights are for the relative contribution of each frequency vector.

    Arguments:

        - freq (``numpy.ndarray``) A ``Prof`` instance or a rank-2 array of
          frequencies along the last dimension.
        - weights (``numpy.ndarray``) An array with a weight for each
          frequency vector. Rank-1.
        - unit (``str``) see: the function ``shannon_entropy``.
    """
    if weights is not None:
        if len(freq) != len(weights):
            raise ValueError('The number of frequencies and weights do not match.')
        if (freq.ndim != 2) or (len(freq) < 2):
            raise ValueError('At least two frequencies in a rank-2 array expected.')
    weighted_average = np.average(freq, axis=0, weights=weights)
    h_avg_freq = shannon_entropy(weighted_average, unit)
    h_freq = shannon_entropy(freq, unit)
    avg_h_freq = np.average(h_freq, weights=weights)
    jsd = h_avg_freq - avg_h_freq
    return jsd


def shannon_entropy(freq, unit='bit'):
    """Calculates the Shannon Entropy (H) of a frequency.

    Arguments:

        - freq (``numpy.ndarray``) A ``Freq`` instance or ``numpy.ndarray`` with
          frequency vectors along the last axis.
        - unit (``str``) The unit of the returned entropy one of 'bit', 'digit'
          or 'nat'.
    """
    log = get_base(unit)
    # keep shape to return in right shape
    shape = freq.shape
    # place to keep entropies
    hs = np.ndarray(freq.size / shape[-1])
    # this returns an array of vectors or just a vector of frequencies
    freq = freq.reshape((-1, shape[-1]))
    # this makes sure we have an array of vectors of frequencies
    freq = np.atleast_2d(freq)
    # get fancy indexing
    positives = freq != 0.
    for i, (freq, idx) in enumerate(itertools.izip(freq, positives)):
        # keep only non-zero
        freq = freq[idx]
        # logarithms of non-zero frequencies
        logs = log(freq)
        hs[i] = -np.sum(freq * logs)
    hs.reshape(shape[:-1])
    return hs


def get_base(unit='bit'):
    if unit == 'bit':
        log = scp.log2
    elif unit == 'nat':
        log = scp.log
    elif unit in ('digit', 'dit'):
        log = scp.log10
    else:
        raise ValueError('The "unit" "%s" not understood' % unit)
    return log
