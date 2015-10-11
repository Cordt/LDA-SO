from gensim import models, utils
from Importer import Importer
from Preprocessor import Preprocessor
from Utilities import *
import Metric
import Reuters
import os
import sys
import sqlite3
import logging


class Topicmodel:

    def __init__(self, setting):
        self.setting = setting
        self.mallet_path = setting['malletpath']
        self.nooftopics = setting['nooftopics']
        self.noofiterations = setting['noofiterations']
        self.model = None
        self.importer = None
        self.preprocessor = None
        self.answer_preprocessor = None
        self.no_of_questions = 0

        # Set tablename for similarities table
        self.sim_tablename = ""
        if self.setting['data_for_model'] == 1:
            # Use only questions
            self.sim_tablename = "question_similarities"

        elif self.setting['data_for_model'] == 2:
            # Use only answers
            self.sim_tablename = "answer_similarities"

        elif self.setting['data_for_model'] == 3:
            # Use both, questions and answers
            self.sim_tablename = "corpus_similarities"

    def __iter__(self):
        for tokens in self.preprocessor.corpus:
            yield self.preprocessor.vocabulary.doc2bow(tokens)

    ####################################################################################################
    # Load data and create model
    ####################################################################################################

    def createmodel(self):
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/model/"
        filename = 'model_' + str(self.setting['data_for_model']) + '.gs'
        path = ''.join([directory, filename])

        try:
            self.model = utils.SaveLoad.load(path)
        except IOError:
            logging.info("Could not load '%s', starting from scratch" % path)

        # Importing data
        self.importer = Importer(self.setting)
        self.importer.import_xml_data()
        self._loadandpreprocess_corpus()

        if self.model is None:
            # Model is learned on questions
            self._learnmodel()
            self._savemodel()

    def _loadandpreprocess_corpus(self):
        if self.setting['theme'] is 'reuters':

            # Use the reuters corpus
            reuters = Reuters.GS('/usr/share/nltk_data/corpora/reuters/training')
            rawdata = reuters.getrawcorpus()

            # Preprocessing data
            self.preprocessor = Preprocessor(self.setting)
            self.preprocessor.create_clean_shortened_vocabulary(rawdata)

        else:
            if self.setting['data_for_model'] == 1:
                # Use only questions
                rawdata = self.importer.get_question_corpus()
                self.preprocessor = Preprocessor(self.setting)
                self.preprocessor.create_clean_shortened_vocabulary(rawdata)

            elif self.setting['data_for_model'] == 2:
                # Use only answers
                rawdata = self.importer.get_answer_corpus()
                self.preprocessor = Preprocessor(self.setting)
                self.preprocessor.create_clean_shortened_vocabulary(rawdata)

            elif self.setting['data_for_model'] == 3:
                # Use both, questions and answers
                question_rawdata = self.importer.get_question_corpus()
                answer_rawdata = self.importer.get_answer_corpus()
                rawdata = []

                for row in question_rawdata:
                    rawdata.append(row)
                for row in answer_rawdata:
                    rawdata.append(row)

                self.preprocessor = Preprocessor(self.setting)
                self.preprocessor.create_clean_shortened_vocabulary(rawdata)

            # We need the number of questions
            self.no_of_questions = self.importer.get_number_of_questions()

            # We need the answer corpus separately
            rawdata = self.importer.get_answer_corpus()
            self.answer_preprocessor = Preprocessor(self.setting)
            self.answer_preprocessor.create_clean_shortened_vocabulary(rawdata)

    def _learnmodel(self):
        self.model = models.wrappers.LdaMallet(self.mallet_path, self, num_topics=self.nooftopics,
                                               id2word=self.preprocessor.vocabulary,
                                               iterations=self.noofiterations)

    def _savemodel(self):
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/model/"
        filename = 'model_' + str(self.setting['data_for_model']) + '.gs'
        path = ''.join([directory, filename])
        if not os.path.exists(directory):
                os.makedirs(directory)
        self.model.save(fname_or_handle=path)

    def _print_and_write_result(self, experiment_no, result):
        output = ''
        output += '\n\n####################\n\n'

        if experiment_no == 1:
            output += 'Experiment #1:\t-\tThe average distance of related answers to a given question\n'
        elif experiment_no == 2:
            output += 'Experiment #2:\t-\tThe average distance of the order of related answers given by the ' \
                      'similarity compared to the actual order\n'
        elif experiment_no == 3:
            output += 'Experiment #3:\t-\tThe average of the impact of the length of an answer to the actual score\n'

        output += 'Setting:\n'
        output += '\tTheme:\t\t\t\t\t\t\t\t\t' + str(self.setting['theme']) + '\n'
        output += '\tNumber of iterations:\t\t\t\t\t' + str(self.setting['noofiterations']) + '\n'

        used_data = ''
        if self.setting['data_for_model'] == 1:
            used_data = 'Questions'
        elif self.setting['data_for_model'] == 2:
            used_data = 'Answers'
        elif self.setting['data_for_model'] == 3:
            used_data = 'Questions and answers'
        output += '\tData used for topic model:\t\t\t\t' + used_data + '\n'

        # Metrics are only used for Experiment 2 and 3
        if experiment_no != 1:
            used_metric = ''
            if self.setting['distance_metric'] == 1:
                used_metric = 'Exact match distance'
            elif self.setting['distance_metric'] == 2:
                used_metric = 'Deviation distance'
            elif self.setting['distance_metric'] == 3:
                used_metric = 'Squared deviaton distance'
            output += '\tDistance metric used for experiment:\t' + used_metric + '\n\n'

        output += 'Result:\t' + str(result) + '\n'
        output += '\n####################\n\n'

        logging.info(output)

        directory = self.setting['resultfolder'] + self.setting['theme'] + "/"
        filename = self.setting['theme'] + '_' + str(experiment_no) + '.txt'
        path = ''.join([directory, filename])

        f = open(path, "a")
        f.write(output)
        f.close()

    def _print_precision_recall_f1(self, precision, recall, f1):
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/"

        data = ""
        if self.setting['data_for_model'] == 1:
            data = "questions"

        elif self.setting['data_for_model'] == 2:
            data = "answers"

        elif self.setting['data_for_model'] == 3:
            data = "qanda"

        precision_filename = self.setting['theme'] + '_precision_' + data + '.txt'
        path = ''.join([directory, precision_filename])

        f = open(path, "w")
        for (index, value) in enumerate(precision):
            f.write(str(index) + ', ' + str(value) + '\n')
        f.close()

        recall_filename = self.setting['theme'] + '_recall_' + data + '.txt'
        path = ''.join([directory, recall_filename])

        f = open(path, "w")
        for (index, value) in enumerate(recall):
            f.write(str(index) + ', ' + str(value) + '\n')
        f.close()

        f1_filename = self.setting['theme'] + '_f1_' + data + '.txt'
        path = ''.join([directory, f1_filename])

        f = open(path, "w")
        for (index, value) in enumerate(f1):
            f.write(str(index) + ', ' + str(value) + '\n')
        f.close()

    ####################################################################################################
    # Calculate statistics on data - Experiment 1, actual distance to question
    ####################################################################################################

    def get_true_answers_distances(self, no_of_questions=-1):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        theme_dbpath = self.setting['dbpath']
        if no_of_questions is -1:
            no_of_questions = get_max_question_id(result_folder_path, self.sim_tablename)
        normalized_distance = 0.0
        average_distance = 0.0

        logging.info('Experiment #1...')

        # We don't count questions that do not have an answer
        actual_no_of_questions = no_of_questions + 1

        number_of_related_answers = None
        if self.setting['minimum_number_of_answers'] > 0:
            # Tuple like (question ID, number of answers)
            number_of_related_answers = get_number_of_related_answers(theme_dbpath)

        for question_id in range(1, no_of_questions + 1):

            # Omit all questions that have less the number of answers then set
            if self.setting['minimum_number_of_answers'] > 0:
                if number_of_related_answers[question_id - 1] < self.setting['minimum_number_of_answers']:
                    actual_no_of_questions -= 1
                    continue

            # Table similarities: [questionId, answerId, similarity], ordered by similarity, ascending

            answer_similarities = load_similarities_for_question(result_folder_path, self.sim_tablename, question_id)

            # Table answer: (id), Ordered By score, descending
            related_answer_ids = get_related_answer_ids(theme_dbpath, question_id, with_score=False)

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
        logging.info('\n\tDone.')

        average_distance /= float(actual_no_of_questions)
        self._print_and_write_result(1, average_distance)

    ####################################################################################################
    # Calculate statistics on data - Experiment 2, distance to order of related questions given by score
    ####################################################################################################

    def compute_answer_order_metric(self, no_of_questions=-1):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        theme_dbpath = self.setting['dbpath']

        logging.info('Experiment #2...')

        average_distance = 0.0

        if no_of_questions == -1:
            no_of_questions = get_max_question_id(result_folder_path, self.sim_tablename)

        actual_no_of_questions = no_of_questions

        number_of_related_answers = None
        if self.setting['minimum_number_of_answers'] > 0:
            # Tuple like (question ID, number of answers)
            number_of_related_answers = get_number_of_related_answers(theme_dbpath)

        for question_id in range(1, no_of_questions + 1):

            # Omit all questions that have less the number of answers then set
            if self.setting['minimum_number_of_answers'] > 0:
                if number_of_related_answers[question_id - 1] < self.setting['minimum_number_of_answers']:
                    actual_no_of_questions -= 1
                    continue

            # Table similarities: (answer ID, similarity), ordered by similarity, ascending
            similarity_ordered_answers = load_related_answer_similarities_for_question(
                theme_dbpath, result_folder_path, self.sim_tablename, question_id)

            # Table answer: (answer ID, score), Ordered By score, descending
            theme_dbpath = self.setting['dbpath']
            score_ordered_answers = get_related_answer_ids(theme_dbpath, question_id, with_score=True)

            distance = 0.0

            # Compute the normalized distance metric
            if self.setting['distance_metric'] == 1:
                distance = Metric.exact_match_distance(score_ordered_answers, similarity_ordered_answers)
            elif self.setting['distance_metric'] == 2:
                distance = Metric.deviation_distance(score_ordered_answers, similarity_ordered_answers)
            elif self.setting['distance_metric'] == 3:
                distance = Metric.squared_deviation_distance(score_ordered_answers, similarity_ordered_answers)

            if distance == -1:
                actual_no_of_questions -= 1
            else:
                average_distance += distance

            # Print percentage of already finished questions
            if question_id % 10 is 0:
                ratio = (float(question_id) / float(no_of_questions)) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()
        logging.info('\n\tDone.')

        average_distance /= float(actual_no_of_questions)
        self._print_and_write_result(2, average_distance)

    ####################################################################################################
    # Calculate statistics on data - Experiment 3, impact of answer length on answer score
    ####################################################################################################

    def compute_answer_length_impact(self, no_of_questions=-1):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        theme_dbpath = self.setting['dbpath']

        logging.info('Experiment #3...')

        average_answer_distance = 0.0

        if no_of_questions == -1:
            no_of_questions = get_max_question_id(result_folder_path, self.sim_tablename)

        actual_no_of_questions = no_of_questions

        number_of_related_answers = None
        if self.setting['minimum_number_of_answers'] > 0:
            # Tuple like (question ID, number of answers)
            number_of_related_answers = get_number_of_related_answers(theme_dbpath)

        for question_id in range(1, no_of_questions + 1):

            # Omit all questions that have less the number of answers then set
            if self.setting['minimum_number_of_answers'] > 0:
                if number_of_related_answers[question_id - 1] < self.setting['minimum_number_of_answers']:
                    actual_no_of_questions -= 1
                    continue

            # Table answer: (answer ID, score), Ordered By score, descending
            score_ordered_answers = get_related_answer_ids(theme_dbpath, question_id, with_score=True)

            # Table lengths: (answer ID, length), Ordered By length, descending
            length_ordered_answers = load_related_answer_lengths(theme_dbpath, result_folder_path, question_id)

            distance = 0.0

            # Compute the normalized distance metric
            if self.setting['distance_metric'] == 1:
                distance = Metric.exact_match_distance(score_ordered_answers, length_ordered_answers)
            elif self.setting['distance_metric'] == 2:
                distance = Metric.deviation_distance(score_ordered_answers, length_ordered_answers)
            elif self.setting['distance_metric'] == 3:
                distance = Metric.squared_deviation_distance(score_ordered_answers, length_ordered_answers)

            if distance == -1:
                actual_no_of_questions -= 1
            else:
                average_answer_distance += distance

            # Print percentage of already finished questions
            if question_id % 10 is 0:
                ratio = (float(question_id) / float(no_of_questions)) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()
        logging.info('\n\tDone.')

        average_answer_distance /= float(actual_no_of_questions)
        self._print_and_write_result(3, average_answer_distance)

    ####################################################################################################
    # Calculate statistics on data - Experiment 4, actual distance to question - precision and recall
    ####################################################################################################

    def get_precision_of_answers_distances(self, no_of_questions=-1):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        theme_dbpath = self.setting['dbpath']
        if no_of_questions is -1:
            no_of_questions = get_max_question_id(result_folder_path, self.sim_tablename)

        logging.info('Experiment #4...')

        precision = []
        recall = []
        f1 = []

        # We don't count questions that do not have an answer
        actual_no_of_questions = no_of_questions

        number_of_related_answers = None
        if self.setting['minimum_number_of_answers'] > 0:
            # Tuple like (question ID, number of answers)
            number_of_related_answers = get_number_of_related_answers(theme_dbpath)

        total_number_of_answers = 0

        for question_id in range(1, no_of_questions + 1):

            # Omit all questions that have less the number of answers then set
            if self.setting['minimum_number_of_answers'] > 0:
                if number_of_related_answers[question_id - 1] < self.setting['minimum_number_of_answers']:
                    actual_no_of_questions -= 1
                    continue

            # Table similarities: [questionId, answerId, similarity], ordered by similarity, ascending
            answer_similarities = load_similarities_for_question(result_folder_path, self.sim_tablename, question_id)
            total_number_of_answers = len(answer_similarities)

            # Table answer: (id), Ordered By score, descending
            theme_dbpath = self.setting['dbpath']
            related_answer_ids = get_related_answer_ids(theme_dbpath, question_id, with_score=False)

            number_of_relevant_answers = 0

            if len(related_answer_ids) is not 0:
                for (answer_index, row) in enumerate(answer_similarities):

                    if len(precision) == answer_index:
                        # Append zeroes to precision and recall
                        precision.append(float(0))
                        recall.append(float(0))
                        f1.append(float(0))

                    # Check whether answer is one of the answers, given to this question
                    if row[1] in related_answer_ids:
                        number_of_relevant_answers += 1

                    precision[answer_index] += (float(number_of_relevant_answers) / float(answer_index + 1))
                    recall[answer_index] += (float(number_of_relevant_answers) / float(len(related_answer_ids)))

            else:
                actual_no_of_questions -= 1

            # Print percentage of already finished questions
            if question_id % 10 is 0:
                ratio = (float(question_id) / float(no_of_questions)) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        # Normalize precision and recall
        for answer_index in range(0, total_number_of_answers):
            precision[answer_index] /= float(actual_no_of_questions)
            recall[answer_index] /= float(actual_no_of_questions)
            if (precision[answer_index] == 0.0) and (recall[answer_index] == 0.0):
                f1[answer_index] = 0.0
            else:
                f1[answer_index] = 2.0 * ((precision[answer_index] * recall[answer_index]) /
                                          (precision[answer_index] + recall[answer_index]))

        logging.info('\n\tDone.')

        self._print_precision_recall_f1(precision, recall, f1)

    ####################################################################################################
    # Calculate and store document similarities
    ####################################################################################################

    def determine_question_answer_distances(self, no_of_answers=-1):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        similarities = []

        # Remove current similarities table, if any
        create_clean_similarities_table(result_folder_path, self.sim_tablename,
                                        self.setting['clean_similarities_table'])

        # Insert data if table not empty
        filename = "/model/similarities.db"
        dbpath = ''.join([result_folder_path, filename])
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        sql = 'SELECT * FROM ' + self.sim_tablename
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()

        if len(result) != 0:
            logging.info('Similarities tables not empty, adding nothing')
            return

        # Load all question topics
        question_topics = []
        for (question_index, question_topic) in enumerate(self.model.load_document_topics(), start=1):
            question_topics.append((question_index, question_topic))
            if question_index >= self.no_of_questions:
                break

        doc_count = 0

        for (answer_index, element) in enumerate(self.answer_preprocessor.corpus, start=1):
            # Determine topics of the answers in the topic model
            bow = self.preprocessor.vocabulary.doc2bow(element)
            answer_topics = self.model[bow]

            for (question_index, question_topic) in question_topics:
                similarities.append((question_index, answer_index, self._compare_documents(question_topic,
                                                                                           answer_topics)))

            write_similarities_to_db(result_folder_path, self.sim_tablename, similarities)
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

    ####################################################################################################
    # Calculate and store document lenghts
    ####################################################################################################

    def determine_answer_lengths(self):
        lengths = []

        # Remove current similarities table, if any
        self._create_clean_lengths_table()

        for (answer_index, element) in enumerate(self.answer_preprocessor.corpus, start=1):
            lengths.append((answer_index, len(element)))

        self._write_lengths_to_db(lengths)

    def _create_clean_lengths_table(self):
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/model/"
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
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/model/"
        filename = "lenghts.db"
        dbpath = ''.join([directory, filename])

        # Database connection - instance variables
        connection = sqlite3.connect(dbpath)
        cursor = connection.cursor()

        for (answer_id, length) in lengths:
            values = [answer_id, length]
            cursor.execute('INSERT INTO lengths VALUES (?, ?)', values)

        connection.commit()
