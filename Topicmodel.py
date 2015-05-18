__author__ = 'Cordt'

from gensim import models, utils
from Importer import Importer
from Preprocessor import Preprocessor
import Similarity as Sim
import Reuters
import os
import sqlite3


class Topicmodel:

    def __init__(self, setting):
        self.setting = setting
        self.mallet_path = setting['malletpath']
        self.nooftopics = setting['nooftopics']
        self.noofiterations = setting['noofiterations']
        self.model = None
        self.importer = None
        self.question_preprocessor = None
        self.answer_preprocessor = None

    def __iter__(self):
        for tokens in self.question_preprocessor.corpus:
            yield self.question_preprocessor.vocabulary.doc2bow(tokens)

    def createmodel(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "model.gs"
        path = ''.join([directory, filename])

        try:
            self.model = utils.SaveLoad.load(path)
        except IOError:
            print("Could not load '%s', starting from scratch" % path)

        # Importing data
        self.importer = Importer(self.setting)
        self.importer.importxmldata()
        self._loadandpreprocess_questions()
        self._loadandpreprocess_answers()

        if self.model is None:
            # Model is learned on questions
            self._learnmodel()
            self._savemodel()

    def determine_document_similarities(self, no_of_documents=3):
        current_document = []
        next_document = []
        similarities = []
        most_similar_documents = []

        # Remove current similarities table, if any
        self._create_clean_similarities_table()

        for doc_count in range(0, no_of_documents):
            for (outer_index, document) in enumerate(self.model.load_document_topics()):
                if (outer_index == 0) and (doc_count == 0):
                    current_document = document
                    continue
                elif outer_index <= doc_count:
                    continue
                if outer_index == (doc_count + 1):
                    next_document = document
                similarities.append((outer_index, self._compare_documents(current_document, document)))

            current_document = next_document

            for (document_id, similarity) in self.sorted_similarity_list(similarities):
                most_similar_documents.append((doc_count, document_id))
                break

            # self._write_similarities_to_db(doc_count, similarities)
            similarities = []

        self._find_most_similar_answers()

    def _find_most_similar_answers(self):
        for element in self.answer_preprocessor.corpus:

            # Determine topics of the answers in the question topic model
            bow = self.question_preprocessor.vocabulary.doc2bow(element)
            topics = self.model[bow]

    def _print_most_similar_documents(self, most_similar_documents):
        dbpath = self.setting['dbpath']

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        for (first_document_id, second_document_id) in most_similar_documents:
            values = [first_document_id + 1]
            cursor.execute('SELECT * FROM id_to_elementId WHERE id=?', values)
            result = cursor.fetchone()
            cursor.execute('SELECT body FROM ' + result[2] + ' WHERE elementId=' + str(result[1]))
            result = cursor.fetchone()
            print('First document:\n%s\n\n' % (result[0]))

            values = [second_document_id + 1]
            cursor.execute('SELECT * FROM id_to_elementId WHERE id=?', values)
            result = cursor.fetchone()
            cursor.execute('SELECT body FROM ' + result[2] + ' WHERE elementId=' + str(result[1]))
            result = cursor.fetchone()
            print('Second document:\n%s\n\n\n\n' % (result[0]))

        connection.commit()

    def _loadandpreprocess_questions(self):
        if self.setting['theme'] is 'reuters':

            # Use the reuters corpus
            reuters = Reuters.GS('/usr/share/nltk_data/corpora/reuters/training')
            rawdata = reuters.getrawcorpus()

            # Preprocessing data
            self.question_preprocessor = Preprocessor(self.setting)
            self.question_preprocessor.simplecleanrawdata(rawdata)

        else:
            rawdata = self.importer.get_question_corpus()

            # Preprocessing data
            self.question_preprocessor = Preprocessor(self.setting)
            self.question_preprocessor.simplecleanrawdata(rawdata)

    def _loadandpreprocess_answers(self):
        rawdata = self.importer.get_answer_corpus()

        # Preprocessing data
        self.answer_preprocessor = Preprocessor(self.setting)
        self.answer_preprocessor.simplecleanrawdata(rawdata)

    def _learnmodel(self):
        self.model = models.wrappers.LdaMallet(self.mallet_path, self, num_topics=self.nooftopics,
                                               id2word=self.question_preprocessor.vocabulary,
                                               iterations=self.noofiterations)

    def _savemodel(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "model.gs"
        path = ''.join([directory, filename])
        if not os.path.exists(directory):
                os.makedirs(directory)
        self.model.save(fname_or_handle=path)

    @staticmethod
    def _compare_documents(first_document, second_document):
        first_topic_description = []
        second_topic_description = []

        for (topic, weight) in first_document:
            first_topic_description.append(weight)

        for (topic, weight) in second_document:
            second_topic_description.append(weight)
        return Sim.jsdistance(first_topic_description, second_topic_description)

    @staticmethod
    def sorted_similarity_list(similarities):
        return sorted(similarities, key=lambda similarity: similarity[1])

    def _create_clean_similarities_table(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        sql = 'DROP TABLE IF EXISTS similarities'
        cursor.execute(sql)

        sql = 'CREATE TABLE IF NOT EXISTS similarities (firstDocumentId int, secondDocumentId int, similarity real)'
        cursor.execute(sql)

    def _write_similarities_to_db(self, first_document_id, similarities):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        for (second_document_id, similarity) in similarities:
            values = [first_document_id, second_document_id, similarity]
            cursor.execute('INSERT INTO similarities VALUES (?, ?, ?)', values)

        connection.commit()


