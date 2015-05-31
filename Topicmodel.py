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

    ####################################################################################################
    # Load data and create model
    ####################################################################################################

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
        self.importer.import_xml_data()
        self._loadandpreprocess_questions()
        self._loadandpreprocess_answers()

        if self.model is None:
            # Model is learned on questions
            self._learnmodel()
            self._savemodel()

    def _loadandpreprocess_questions(self):
        if self.setting['theme'] is 'reuters':

            # Use the reuters corpus
            reuters = Reuters.GS('/usr/share/nltk_data/corpora/reuters/training')
            rawdata = reuters.getrawcorpus()

            # Preprocessing data
            self.question_preprocessor = Preprocessor(self.setting)
            self.question_preprocessor.simple_clean_raw_data(rawdata)

        else:
            rawdata = self.importer.get_question_corpus()

            # Preprocessing data
            self.question_preprocessor = Preprocessor(self.setting)
            self.question_preprocessor.simple_clean_raw_data(rawdata)

    def _loadandpreprocess_answers(self):
        rawdata = self.importer.get_answer_corpus()

        # Preprocessing data
        self.answer_preprocessor = Preprocessor(self.setting)
        self.answer_preprocessor.simple_clean_raw_data(rawdata)

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

    ####################################################################################################
    # Calculate statistics on data
    ####################################################################################################

    def determine_question_answer_distances(self, no_of_answers=-1):
        similarities = []

        # Remove current similarities table, if any
        self._create_clean_similarities_table()

        doc_count = 0

        for (answer_index, element) in enumerate(self.answer_preprocessor.corpus):
            # Determine topics of the answers in the question topic model
            bow = self.question_preprocessor.vocabulary.doc2bow(element)
            answer_topics = self.model[bow]

            for (question_index, question_topics) in enumerate(self.model.load_document_topics()):
                similarities.append((question_index, answer_index, self._compare_documents(question_topics,
                                                                                           answer_topics)))

            self._write_similarities_to_db(similarities)
            similarities = []

            doc_count += 1
            if doc_count == no_of_answers:
                break

    def get_true_answers_distances(self, no_of_questions=-1):
        if no_of_questions is -1:
            no_of_questions = self._get_max_question_id()
        normalized_distance = 0.0
        average_distance = 0.0
        for question_id in range(0, no_of_questions):
            answer_similarities = self._load_similarities_for_question(question_id)
            related_answer_ids = self._get_related_answer_ids(question_id)
            if len(related_answer_ids) is not 0:
                for (answer_index, row) in enumerate(answer_similarities):
                    # Check whether answer is one of the answers, given to this question
                    if row[1] in related_answer_ids:
                        normalized_distance += (float(answer_index) / float(len(answer_similarities)))

                normalized_distance /= float(len(related_answer_ids))
                average_distance += normalized_distance
                normalized_distance = 0.0

        average_distance /= float(no_of_questions)
        print(average_distance)

    def _load_similarities_for_question(self, question_id):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        values = [question_id]
        cursor.execute('SELECT * FROM `similarities` WHERE questionId=? ORDER BY similarity DESC', values)

        return cursor.fetchall()

    def _get_max_question_id(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        cursor.execute('SELECT MAX(questionId) FROM similarities')
        return cursor.fetchone()[0]

    def _get_related_answer_ids(self, question_id):
        dbpath = self.setting['dbpath']

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        answer_ids = []

        # Get the original question ID
        # Document counting starts at 1
        values = [question_id + 1]
        cursor.execute('SELECT elementId FROM id_to_question_elementId WHERE id=?', values)
        question_element_id = cursor.fetchone()[0]

        # Get the related answers
        values = [question_element_id]
        cursor.execute('SELECT elementId FROM answer WHERE questionId=?', values)
        related_answer_element_ids = cursor.fetchall()

        # Translate answer element ID's to ID's
        for answer_element_id in related_answer_element_ids:
            values = [answer_element_id[0]]
            cursor.execute('SELECT id FROM id_to_answer_elementId WHERE elementId=?', values)
            answer_ids.append(cursor.fetchone()[0])

        return answer_ids

    ####################################################################################################
    # Calculate and store document similarities
    ####################################################################################################

    @staticmethod
    def _compare_documents(first_document, second_document):
        first_topic_description = []
        second_topic_description = []

        for (topic, weight) in first_document:
            first_topic_description.append(weight)

        for (topic, weight) in second_document:
            second_topic_description.append(weight)
        return Sim.js_distance(first_topic_description, second_topic_description)

    def _create_clean_similarities_table(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        sql = 'DROP TABLE IF EXISTS similarities'
        cursor.execute(sql)

        sql = 'CREATE TABLE IF NOT EXISTS similarities (questionId int, answerId int, similarity real)'
        cursor.execute(sql)

    def _write_similarities_to_db(self, similarities):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        for (question_id, answer_id, similarity) in similarities:
            values = [question_id, answer_id, similarity]
            cursor.execute('INSERT INTO similarities VALUES (?, ?, ?)', values)

        connection.commit()

    @staticmethod
    def sorted_similarity_list(similarities):
        return sorted(similarities, key=lambda similarity: similarity[2])
