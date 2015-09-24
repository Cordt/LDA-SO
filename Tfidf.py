from Importer import Importer
from Preprocessor import Preprocessor
from numpy import zeros
from numpy import dot
from numpy.linalg import norm
from Utilities import *
import logging
import sys
import math


class TFIDF:

    def __init__(self, setting):
        self.setting = setting
        self.importer = None
        self.question_preprocessor = None
        self.answer_preprocessor = None
        self.question_vectors = None
        self.answer_vectors = None
        self.no_of_questions = 0
        self.idf_values = None
        self.sim_tablename = "tf_idf"

        # Importe data
        self.importer = Importer(self.setting)
        self.importer.import_xml_data()
        # Preprocess data
        self._loadandpreprocess_corpus()

    def _print_precision_recall(self, precision, recall):
        directory = self.setting['resultfolder'] + self.setting['theme'] + "/"

        precision_filename = self.setting['theme'] + '_precision_tf-idf.txt'
        path = ''.join([directory, precision_filename])

        f = open(path, "w")
        for (index, value) in enumerate(precision):
            f.write(str(index) + ', ' + str(value) + '\n')
        f.close()

        recall_filename = self.setting['theme'] + '_recall_tf-idf.txt'
        path = ''.join([directory, recall_filename])

        f = open(path, "w")
        for (index, value) in enumerate(recall):
            f.write(str(index) + ', ' + str(value) + '\n')
        # f.close()

    ####################################################################################################
    # Load data and preprocess corpus
    ####################################################################################################

    def _loadandpreprocess_corpus(self):
        rawdata = self.importer.get_question_corpus()
        self.question_preprocessor = Preprocessor(self.setting)
        self.question_preprocessor.create_clean_corpus(rawdata)

        rawdata = self.importer.get_answer_corpus()
        self.answer_preprocessor = Preprocessor(self.setting)
        self.answer_preprocessor.create_clean_corpus(rawdata)

        # We need the number of questions
        self.no_of_questions = self.importer.get_number_of_questions()

    ####################################################################################################
    # Calculate statistics on data - Experiment 5, IR as in #1, but using TF-IDF
    ####################################################################################################

    def get_tf_idf_precision_recall(self, no_of_questions):
        result_folder_path = self.setting['resultfolder'] + self.setting['theme']
        theme_dbpath = self.setting['dbpath']
        if no_of_questions is -1:
            # IDs start from 1, so we need to add 1 to get the total number of questions
            no_of_questions = get_max_question_id(result_folder_path, self.sim_tablename)

        logging.info('Experiment #5...')

        precision = []
        recall = []

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

            # Table similarities: [questionId, answerId, similarity], ordered by similarity, DESCENDING
            answer_similarities = load_similarities_for_question(result_folder_path, self.sim_tablename, question_id,
                                                                 False)
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

        logging.info('\n\tDone.')

        self._print_precision_recall(precision, recall)

    def determine_tf_idf_similarities(self):
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

        # Calculate all TF-IDF vectors
        self._get_tf_idf_vectors()

        logging.info('\nDetermining question-answer similartites.')

        for (answer_index, answer_vector) in enumerate(self.answer_vectors):
            for (question_index, question_vector) in enumerate(self.question_vectors):

                # Cosine similartity
                sim = float(dot(question_vector, answer_vector) / (norm(question_vector) * norm(answer_vector)))
                similarities.append((question_index + 1, answer_index + 1, sim))

            write_similarities_to_db(result_folder_path, self.sim_tablename, similarities)
            similarities = []

            # Print percentage of already finished similartities
            if answer_index % 10 is 0:
                ratio = (float(answer_index) / float(len(self.answer_vectors))) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        logging.info('\nDone.')

    def _get_tf_idf_vectors(self):
        q_corpus = self.question_preprocessor.corpus
        a_corpus = self.answer_preprocessor.corpus
        corpus = q_corpus + a_corpus
        dictionary = Preprocessor.get_dictionary_from_corpora([q_corpus, a_corpus])

        self.question_vectors = []
        self.answer_vectors = []

        self.idf_values = zeros(len(dictionary))

        logging.info('\nDetermining question vectors.')

        for question_index, question in enumerate(q_corpus):
            question_vector = zeros(len(dictionary))
            for index, word in enumerate(dictionary.token2id):
                question_vector[index] = self.tfidf(word, index, question, corpus)
            self.question_vectors.append(question_vector)

            # Print percentage of already finished questions
            if question_index % 10 is 0:
                ratio = (float(question_index) / float(len(q_corpus))) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        print(len(self.question_vectors))

        logging.info('\n\tDone.')

        logging.info('\nDetermining answer vectors.')

        for answer_index, question in enumerate(a_corpus):
            answer_vector = zeros(len(dictionary))
            for index, word in enumerate(dictionary.token2id):
                answer_vector[index] = self.tfidf(word, index, question, corpus)
            self.answer_vectors.append(answer_vector)

            # Print percentage of already finished questions
            if answer_index % 10 is 0:
                ratio = (float(answer_index) / float(len(a_corpus))) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        logging.info('\n\tDone.')

    @staticmethod
    def tf(word, document):
        wordcount = 0
        for current_word in document:
            if current_word == word:
                wordcount += 1
        return float(wordcount) / len(document)

    @staticmethod
    def n_containing_documents(word, corpus):
        return sum(1 for document in corpus if word in document)

    @staticmethod
    def idf(word, documents):
        return math.log(len(documents) / (TFIDF.n_containing_documents(word, documents)))

    def tfidf(self, word, word_index, document, corpus):
        tf = TFIDF.tf(word, document)
        if tf == 0.0:
            return 0.0

        if self.idf_values[word_index] == 0.0:
            self.idf_values[word_index] = TFIDF.idf(word, corpus)
        return tf * self.idf_values[word_index]
