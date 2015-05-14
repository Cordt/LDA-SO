__author__ = 'Cordt'

from Utilities import *
import numpy as np
import sys
from multiprocessing import Process, Lock, Queue
from gensim import corpora, utils


class Preprocessor:

    def __init__(self, setting):
        self.setting = setting
        self.noofprocesses = self.setting['noofprocesses']

        self.corpus = []
        self.vocabulary = set
        self.doctermmatrix = np.array
        self.lock = Lock()

    def simplecleanrawdata(self, rawdata):
        print("Cleaning data...")
        for row in rawdata:
            tmptext = stripcodeblocks(row)
            if self.setting['theme'] != 'reuters':
                tmptext = removetags(tmptext)
            self.corpus.append(utils.simple_preprocess(tmptext))

        print("Removing short words...")
        for (index, document) in enumerate(self.corpus):
            self.corpus[index] = removeshortwords(document)

        print("Retrieving vocabulary...")
        self.vocabulary = corpora.Dictionary(self.corpus)

        print("Removing very common and very uncommon words...")
        self.vocabulary.filter_extremes()

    def createcleandata(self, rawdata):
        print("Cleaning data...")
        self._cleandata(rawdata)

        print("Removing very common and very uncommon words (cut of %d%% each side)" % self.setting['histogramcut'])
        self.corpus = removealphacut(self.corpus, self.setting['histogramcut'])

        print("Retrieving vocabulary...")
        self._setvocabulary()

        print("Creating document x term matrix...")
        self._setdoctermmatrix()

    def _cleandata(self, rawdata):
        # Text cleaning
        for row in rawdata:
            tmptext = stripcodeblocks(row)
            if self.setting['theme'] != 'reuters':
                tmptext = removetags(tmptext)
            tmptext = tmptext.lower()
            tokens = removepunctuation(tmptext)
            tokens = removeshortwords(tokens)
            tokens = removestopwords(tokens)
            tokens = stemtokens(tokens)

            # Append document to corpus
            self.corpus.append(tokens)

    def _setvocabulary(self):
        wordlist = []

        # Append tokens to word list
        for document in self.corpus:
            for token in document:
                wordlist.append(token)

        self.vocabulary = tuple(set(wordlist))

    def getvocabulary(self):
        return self.vocabulary

    def _setdoctermmatrix(self):
        self.doctermmatrix = np.zeros((len(self.corpus), len(self.vocabulary)), dtype=np.int)

        iterations = len(self.corpus) * len(self.vocabulary)
        print('\tNumber of documents: %d, Vocabulary length: %d' % (len(self.corpus), len(self.vocabulary)))
        print('\tNecessary comparisons: %d' % iterations)

        # Using several threads
        docsperthread = len(self.corpus) / self.noofprocesses

        processes = []
        doctermcount = Counter(0)
        resultqueue = Queue()

        if docsperthread > 1:

            # Create processes
            for processindex in range(0, self.noofprocesses):
                endindex = docsperthread
                documentset = np.zeros((docsperthread, len(self.vocabulary)), dtype=int)
                process = Process(target=self._writematrix,
                                  args=[resultqueue, processindex, self.corpus, documentset, self.vocabulary,
                                        doctermcount, endindex, iterations,
                                        (1.0 / float(self.noofprocesses))])
                process.start()
                processes.append(process)

            # Create on more process for the remainder of the documents
            if len(self.corpus) % self.noofprocesses is not 0:
                endindex = len(self.corpus) % self.noofprocesses
                documentset = np.zeros((endindex, len(self.vocabulary)))
                process = Process(target=self._writematrix,
                                  args=[resultqueue, self.noofprocesses, self.corpus, documentset, self.vocabulary,
                                        doctermcount, endindex, iterations,
                                        (1.0 / float(self.noofprocesses))])
                process.start()
                processes.append(process)

        else:
            documentset = np.zeros((len(self.corpus), len(self.vocabulary)), dtype=int)
            self._writematrix(resultqueue, 0, self.corpus, documentset, self.vocabulary, doctermcount,
                              len(self.corpus), iterations, (1.0 / float(self.noofprocesses)))

        count = 0
        if len(self.corpus) % self.noofprocesses is not 0:
            # We need one extra round
            count = -1
        while count is not self.noofprocesses:

            # Combine the results in one doctermmatrix
            result = resultqueue.get()
            startindex = result[0] * docsperthread
            if result[0] is not self.noofprocesses:
                endindex = startindex + docsperthread
            else:
                endindex = startindex + (len(self.corpus) % self.noofprocesses)
            for index in range(startindex, endindex):
                self.doctermmatrix[index] = result[1][index - startindex]
            count += 1

        for process in processes:
            process.terminate()

        sys.stdout.write("\r\t100%")
        sys.stdout.flush()
        print

    @staticmethod
    def _writematrix(resultqueue, processindex, corpus, documentset, vocabulary, doctermcount, enddoc,
                     iterations, docshare):
        count = 0
        lastcount = 0
        # Create incidence matrix
        for docindex in range(0, enddoc):
            for tokenindex, token in enumerate(vocabulary):
                if token in corpus[docindex]:
                    documentset[docindex][tokenindex] = 1
                count += 1
                if count % (10000 + (10000 * docshare)) == 0:
                    doctermcount.increment(by=(count - lastcount))
                    lastcount = count
                    ratio = (float(doctermcount.value()) / float(iterations)) * 100.0
                    sys.stdout.write("\r\t%d%%" % ratio)
                    sys.stdout.flush()
        resultqueue.put((processindex, documentset))

    def getdoctermmatrix(self):
        return self.doctermmatrix