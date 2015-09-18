from Importer import Importer
from Preprocessor import Preprocessor
from numpy import zeros
from numpy import dot
from numpy.linalg import norm
import logging
import sys
import math


class TFIDF:

    def __init__(self, setting):
        self.setting = setting
        self.importer = None
        self.question_preprocessor = None
        self.answer_preprocessor = None
        self.no_of_questions = 0
        self.idf_values = None

        # Importe data
        self.importer = Importer(self.setting)
        self.importer.import_xml_data()
        # Preprocess data
        self._loadandpreprocess_corpus()

    ####################################################################################################
    # Load data and create model
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

    def get_tf_idf_precision_recall(self):
        self.get_tf_idf_score()

    def get_tf_idf_score(self):
        q_corpus = self.question_preprocessor.corpus
        a_corpus = self.answer_preprocessor.corpus
        corpus = q_corpus + a_corpus
        dictionary = Preprocessor.get_dictionary_from_corpora([q_corpus, a_corpus])

        question_vectors = []
        answer_vectors= []

        self.idf_values = zeros(len(dictionary))

        logging.info('\nDetermining question vectors.')

        for question_index, question in enumerate(q_corpus):
            question_vector = zeros(len(dictionary))
            for index, word in enumerate(dictionary.token2id):
                question_vector[index] = self.tfidf(word, index, question, corpus)
            question_vectors.append(question_vector)
            if question_index == 10:
                break
            # Print percentage of already finished questions
            if question_index % 10 is 0:
                ratio = (float(question_index) / float(len(q_corpus))) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        logging.info('\n\tDone.')

        logging.info('\nDetermining answer vectors.')

        for answer_index, question in enumerate(a_corpus):
            answer_vector = zeros(len(dictionary))
            for index, word in enumerate(dictionary.token2id):
                answer_vector[index] = self.tfidf(word, index, question, corpus)
            question_vectors.append(answer_vector)
            if answer_index == 10:
                break
            # Print percentage of already finished questions
            if answer_index % 10 is 0:
                ratio = (float(answer_index) / float(len(a_corpus))) * 100.0
                sys.stdout.write("\r\t%d%%" % ratio)
                sys.stdout.flush()

        logging.info('\n\tDone.')

        logging.info('\nDetermining question-answer similartites.')

        for question_vector in question_vectors:
            for answer_vector in answer_vectors:
                # Cosine similartity
                float(dot(question_vector, answer_vector) / (norm(question_vector) * norm(answer_vector)))

        logging.info('\nDone.')

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
