__author__ = 'Cordt'

from HTMLParser import HTMLParser
from BeautifulSoup import *
from nltk import data
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from multiprocessing import Value, Lock
import numpy as np
import matplotlib.pyplot as plt
import operator


def stripcodeblocks(text):
    result = text
    asoup = BeautifulSoup(result)
    codeblocks = asoup.findAll('code')
    for block in codeblocks:
        block.extract()
    return result


def removestopwords(tokens):
    cachedstopwords = stopwords.words("english")
    result = []
    for token in tokens:
        if token not in cachedstopwords:
            result.append(token)
    return result


def removepunctuation(text):
    result = []
    sentencedetector = data.load('tokenizers/punkt/english.pickle')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens1 = tokenizer.tokenize(text)
    for token1 in tokens1:
        for token2 in sentencedetector.tokenize(token1):
            result.append(token2)
    return result


def removeshortwords(tokens):
    result = []
    for token in tokens:
        if len(token) > 2:
            result.append(token)
    return result


def stemtokens(tokens):
    result = tokens
    porterstemmer = PorterStemmer()
    count = 0
    for token in result:
        result[count] = porterstemmer.stem(token)
        count += 1
    return result


def appendhistogram(histogram, document):
    vocabulary = set(document)
    for token in vocabulary:
        if token in histogram:
            histogram[token] += 1
        else:
            histogram[token] = 1
    return histogram


def printhistogram(histogram):
    sortedhistogram = sorted(histogram.items(), key=operator.itemgetter(1))
    shape = np.arange(len(sortedhistogram))
    getter = operator.itemgetter(1)
    sortedvalues = map(getter, sortedhistogram)
    plt.bar(shape, sortedvalues, align='center', width=0.5)
    plt.xticks(shape, [])
    ymax = max(sortedvalues) + 1
    plt.ylim(0, ymax)
    plt.show()


def removealphacut(corpus, cut):
    histogram = dict()
    for document in corpus:
        histogram = appendhistogram(histogram, document)
    integral = 0
    for value in histogram.values():
        integral += value

    lowerbound = int((float(integral) / 100.0) * cut)
    upperbound = int((float(integral) / 100.0) * (100.0 - cut))

    sortedhistogram = sorted(histogram.items(), key=operator.itemgetter(1))
    getter = operator.itemgetter(1)
    sortedvalues = map(getter, sortedhistogram)

    lowerincidence = 0
    upperincidence = 0
    tmp = 0
    for value in sortedvalues:
        tmp += value
        if tmp > lowerbound:
            lowerincidence = value
            break

    tmp = 0
    for value in reversed(sortedvalues):
        tmp += value
        if tmp > (integral - upperbound):
            upperincidence = value
            break

    newcorpus = []
    for document in corpus:
        newdocument = []
        for token in document:
            if lowerincidence < histogram[token] <= upperincidence:
                newdocument.append(token)
        newcorpus.append(newdocument)

    return newcorpus


class MLStripper(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, markup):
        self.fed.append(markup)

    def get_data(self):
        return ''.join(self.fed)


def removetags(text):
    s = MLStripper()
    s.feed(text)
    return s.get_data()


class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self, by=1):
        with self.lock:
            self.val.value += by

    def value(self):
        with self.lock:
            return self.val.value