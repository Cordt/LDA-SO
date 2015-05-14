__author__ = 'Cordt'

from gensim import models, utils
from Importer import Importer
from Preprocessor import Preprocessor
import Reuters
import os


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