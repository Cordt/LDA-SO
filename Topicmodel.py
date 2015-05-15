__author__ = 'Cordt'

from gensim import models, utils
from Importer import Importer
from Preprocessor import Preprocessor
import Similarity as Sim
import Reuters
import os
import numpy as np


class Topicmodel:

    def __init__(self, setting):
        self.setting = setting
        self.mallet_path = setting['malletpath']
        self.nooftopics = setting['nooftopics']
        self.noofiterations = setting['noofiterations']
        self.model = None
        self.preprocessor = None

    def __iter__(self):
        for tokens in self.preprocessor.corpus:
            yield self.preprocessor.vocabulary.doc2bow(tokens)

    def createmodel(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "model.gs"
        path = ''.join([directory, filename])

        try:
            self.model = utils.SaveLoad.load(path)
        except IOError:
            print("Could not load '%s', starting from scratch" % path)

        if self.model is None:
            self._loadandpreprocess()
            self._learnmodel()
            self._savemodel()

    def compare_documents(self):
        no_of_comparisons = 4
        documents = []
        doc_array = []
        for index in range(0, no_of_comparisons):
            doc_array.append(np.ndarray(shape=(2, self.nooftopics), dtype=float))
        doc_list = []

        for (outer_index, document) in enumerate(self.model.load_document_topics()):
            if outer_index == no_of_comparisons:
                break

            doc_list.append([])
            documents.append(document)
            for (inner_index, (topic, weight)) in enumerate(document):
                doc_list[outer_index].append(weight)
                if outer_index == 0:
                    for document_index in range(0, no_of_comparisons):
                        doc_array[document_index][0][inner_index] = weight
                doc_array[outer_index][1][inner_index] = weight

        print("\nFirst version")
        first_doc = None
        for (index, document) in enumerate(doc_list):
            if index == 0:
                first_doc = document
                continue
            print(2 * Sim.jsdivergence(first_doc, document))

        print("\nSecond version")
        for (index, document) in enumerate(documents):
            if index == 0:
                first_doc = document
                continue
            print(Sim.jsd(first_doc, document, num_topics=self.nooftopics))

        print("\nThird version")
        for (index, document) in enumerate(doc_array):
            if index == 0:
                continue
            print(Sim.jensen_shannon_divergence(document)[0])

    def _loadandpreprocess(self):
        if self.setting['theme'] is 'reuters':

            # Use the reuters corpus
            reuters = Reuters.GS('/usr/share/nltk_data/corpora/reuters/training')
            rawdata = reuters.getrawcorpus()

            # Preprocessing data
            self.preprocessor = Preprocessor(self.setting)
            self.preprocessor.simplecleanrawdata(rawdata)

        else:
            # Importing data
            importer = Importer(self.setting)
            rawdata = importer.importxmldata()

            # Preprocessing data
            self.preprocessor = Preprocessor(self.setting)
            self.preprocessor.simplecleanrawdata(rawdata)

    def _learnmodel(self):
        self.model = models.wrappers.LdaMallet(self.mallet_path, self, num_topics=self.nooftopics,
                                               id2word=self.preprocessor.vocabulary, iterations=self.noofiterations)

    def _savemodel(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "model.gs"
        path = ''.join([directory, filename])
        if not os.path.exists(directory):
                os.makedirs(directory)
        self.model.save(fname_or_handle=path)