__author__ = 'Cordt'

from gensim import corpora, utils
from HTMLParser import HTMLParser
from BeautifulSoup import *


class Preprocessor:

    def __init__(self, setting):
        self.setting = setting

        self.corpus = []
        self.vocabulary = set

    def simple_clean_raw_data(self, raw_data):
        logging.info("Cleaning data...")
        for row in raw_data:
            tmp_text = self.strip_code_blocks(row)
            if self.setting['theme'] != 'reuters':
                tmp_text = MLStripper.remove_tags(tmp_text)
            self.corpus.append(utils.simple_preprocess(tmp_text))

        logging.info("Removing short words...")
        for (index, document) in enumerate(self.corpus):
            self.corpus[index] = self.remove_short_words(document)

        logging.info("Retrieving vocabulary...")
        self.vocabulary = corpora.Dictionary(self.corpus)

        logging.info("Removing very common and very uncommon words...")
        no_below = self.setting['filter_less_than_no_of_documents']
        no_above = self.setting['filter_more_than_fraction_of_documents']
        self.vocabulary.filter_extremes(no_below=no_below, no_above=no_above)

    @staticmethod
    def strip_code_blocks(text):
        result = text
        soup = BeautifulSoup(result)
        code_blocks = soup.findAll('code')
        for block in code_blocks:
            block.extract()
        return result

    @staticmethod
    def remove_short_words(tokens):
        result = []
        for token in tokens:
            if len(token) > 2:
                result.append(token)
        return result


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

    @staticmethod
    def remove_tags(text):
        s = MLStripper()
        s.feed(text)
        return s.get_data()
