__author__ = 'Cordt'

from gensim import models, utils
from Importer import Importer
from Preprocessor import Preprocessor
import Metric
import Reuters
import os
import sys
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
    # Calculate statistics on data - Experiment 1, actual distance to question
    ####################################################################################################

    def get_true_answers_distances(self, no_of_questions=-1):
        if no_of_questions is -1:
            no_of_questions = self._get_max_question_id()
        normalized_distance = 0.0
        average_distance = 0.0

        print('Calculating distance of related answers to a given question...')

        # We don't count questions that do not have an answer
        actual_no_of_questions = no_of_questions + 1

        for question_id in range(1, no_of_questions + 1):

            # Table similarities: [questionId, answerId, similarity], ordered by similarity, ascending
            answer_similarities = self._load_similarities_for_question(question_id)

            # Table answer: (id), Ordered By score, descending
            related_answer_ids = self._get_related_answer_ids(question_id, with_score=False)

            # The actual_answer_index and the actual_total_number_of_answers make sure that the
            # metric is normalized from 0 to 1
            actual_answer_index = 0
            actual_total_number_of_answers = len(answer_similarities) - 1

            if len(related_answer_ids) is not 0:
                for row in answer_similarities:

                    # Check whether answer is one of the answers, given to this question
                    if row[1] in related_answer_ids:
                        # We need to substract 1 in order to normalize to 0
                        normalized_distance += (float(actual_answer_index) / float(actual_total_number_of_answers))

                        actual_answer_index -= 1
                        actual_total_number_of_answers -= 1
                    actual_answer_index += 1

                normalized_distance /= float(len(related_answer_ids))
                average_distance += normalized_distance
                normalized_distance = 0.0

            else:
                actual_no_of_questions -= 1

            # Print percentage of already finished questions
            if question_id % 10 is 0:
                ratio = (float(question_id) / float(no_of_questions)) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()
        print('\n\tDone.')

        average_distance /= float(actual_no_of_questions)
        print('The average distance of related questions to a given question is %s' % average_distance)

    ####################################################################################################
    # Calculate statistics on data - Experiment 2
    ####################################################################################################

    def compute_answer_order_metric(self, no_of_questions=-1):

        print('Calculating the average edit distance of answer, related to a question, with respect to their'
              'score and similarity measure...')

        average_deviation_distance = 0.0

        if no_of_questions == -1:
            no_of_questions = self._get_max_question_id()

        actual_no_of_questions = no_of_questions

        for question_id in range(1, no_of_questions + 1):

            # Table similarities: (answer ID, similarity), ordered by similarity, ascending
            similarity_ordered_answers = self._load_related_answer_similarities_for_question(question_id)

            # Table answer: (answer ID, score), Ordered By score, descending
            score_ordered_answers = self._get_related_answer_ids(question_id, with_score=True)

            # Compute the normalized deviation distance
            deviation_distance = Metric.deviation_distance(score_ordered_answers, similarity_ordered_answers)
            if deviation_distance == -1:
                actual_no_of_questions -= 1
            else:
                average_deviation_distance += deviation_distance

            # Print percentage of already finished questions
            if question_id % 10 is 0:
                ratio = (float(question_id) / float(no_of_questions)) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()
        print('\n\tDone.')

        average_deviation_distance /= float(actual_no_of_questions)
        print('The average deviation distance of answers related to a question is %s' % average_deviation_distance)

    ####################################################################################################
    # Calculate statistics on data - Experiment 3
    ####################################################################################################

    ####################################################################################################
    # Calculate statistics on data - Helper methods
    ####################################################################################################

    def _load_similarities_for_question(self, question_id):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        values = [question_id]
        # The smaller the closer --> ascending
        cursor.execute('SELECT questionId, answerId, similarity FROM `similarities` WHERE questionId=? '
                       'ORDER BY similarity ASC', values)

        return cursor.fetchall()

    def _load_related_answer_similarities_for_question(self, question_id):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        related_answers_ids = self._get_related_answer_ids(question_id, with_score=False)
        related_answers_id_strings = []
        for element in related_answers_ids:
            related_answers_id_strings.append(str(element))
        tmp_string = ','.join(related_answers_id_strings)

        # The smaller the closer --> ascending
        cursor.execute('SELECT answerId, similarity FROM `similarities` WHERE questionId=' + str(question_id) +
                       ' AND answerId IN (' + tmp_string + ') ORDER BY similarity ASC')

        result = cursor.fetchall()
        related_answer_similarities = []
        for row in result:
            # Store a tuple like (answer ID, similarity)
            related_answer_similarities.append((row[0], row[1]))

        return related_answer_similarities

    def _get_max_question_id(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        cursor.execute('SELECT MAX(questionId) FROM similarities')
        return cursor.fetchone()[0]

    def _get_related_answer_ids(self, question_id, with_score=False):
        dbpath = self.setting['dbpath']

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        answer_ids = []

        # Get the related answers and their scores
        values = [question_id]
        cursor.execute('SELECT id, score FROM answer WHERE questionId=? ORDER BY score DESC', values)
        related_answer_element_ids = cursor.fetchall()

        # Translate answer element ID's to ID's
        for answer_element_id in related_answer_element_ids:
            if with_score:
                # Store a tuple like (answer ID, score)
                answer_ids.append((answer_element_id[0], answer_element_id[1]))
            else:
                answer_ids.append(answer_element_id[0])

        return answer_ids

    ####################################################################################################
    # Calculate and store document similarities
    ####################################################################################################

    def determine_question_answer_distances(self, no_of_answers=-1):
        similarities = []

        # Remove current similarities table, if any
        self._create_clean_similarities_table()

        doc_count = 0

        for (answer_index, element) in enumerate(self.answer_preprocessor.corpus, start=1):
            # Determine topics of the answers in the question topic model
            bow = self.question_preprocessor.vocabulary.doc2bow(element)
            answer_topics = self.model[bow]

            for (question_index, question_topics) in enumerate(self.model.load_document_topics(), start=1):
                similarities.append((question_index, answer_index, self._compare_documents(question_topics,
                                                                                           answer_topics)))

            self._write_similarities_to_db(similarities)
            similarities = []

            doc_count += 1
            if doc_count == no_of_answers:
                break

    @staticmethod
    def _compare_documents(first_document, second_document):
        first_topic_description = []
        second_topic_description = []

        for (topic, weight) in first_document:
            first_topic_description.append(weight)

        for (topic, weight) in second_document:
            second_topic_description.append(weight)
        return Metric.js_distance(first_topic_description, second_topic_description)

    def _create_clean_similarities_table(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "similarities.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        sql = 'DROP TABLE IF EXISTS similarities'
        cursor.execute(sql)

        sql = 'CREATE TABLE IF NOT EXISTS similarities (questionId int, answerId int, similarity real, ' \
              'PRIMARY KEY (questionId, answerId))'
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

    ####################################################################################################
    # Calculate and store document lenghts
    ####################################################################################################

    def _create_clean_length_table(self):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "lenghts.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        sql = 'DROP TABLE IF EXISTS lengths'
        cursor.execute(sql)

        sql = 'CREATE TABLE IF NOT EXISTS lengths (answerId INTEGER PRIMARY KEY, length int)'
        cursor.execute(sql)

    def _write_lengths_to_db(self, lengths):
        directory = "../../results/" + self.setting['theme'] + "/model/"
        filename = "lenghts.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        for (answer_id, length) in lengths:
            values = [answer_id, length]
            cursor.execute('INSERT INTO lengths VALUES (?, ?)', values)

        connection.commit()
