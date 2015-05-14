__author__ = 'Cordt Voigt'


from Importer import Importer
from Preprocessor import Preprocessor
from Topicmodel import Topicmodel
import logging
from gensim import utils
import sqlite3
import Reuters
from Tagcloud import Tagcloud

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

theme = 'beer'
setting = {
    'theme': theme,
    'dbpath': '../../data/' + theme + '.db',
    'nooftopics': 10,
    'noofwordsfortopic': 20,
    'noofiterations': 100,
    'noofprocesses': 20,
    'histogramcut': 20.0}

if theme is 'reuters':
    setting['folderprefix'] = '../../data/' + theme + '/'
else:
    setting['folderprefix'] = '../../data/' + theme + '.stackexchange.com/'


if theme is 'reuters':
    # rextractor = Reuters.Importer(setting)
    # corpus = rextractor.importdata()

    # Use the reuters corpus
    reuters = Reuters.GS('/usr/share/nltk_data/corpora/reuters/training')
    rawdata = reuters.getrawcorpus()

    # Preprocessing data
    prepro = Preprocessor(setting)
    prepro.simplecleanrawdata(rawdata)

else:
    # Importing data
    importer = Importer(setting)
    rawdata = importer.importxmldata()

    # Preprocessing data
    prepro = Preprocessor(setting)
    prepro.simplecleanrawdata(rawdata)

    vocabulary = prepro.getvocabulary()
    doctermmatrix = prepro.getdoctermmatrix()


# Topic model
tm = Topicmodel(prepro, setting)
count = 0

for topic in tm.model.show_topics(num_topics=-1, num_words=setting['noofwordsfortopic'], formatted=False):
    words = []
    t = Tagcloud()
    for (prob, word) in topic:
        words.append({"text": word, "weight": prob})
    filename = "../../results/Reuters/Topic-" + str(count) + ".jpg"
    print t.draw(words, imagefilepath=filename)
    count += 1

# Database connection
# connection = sqlite3.connect(setting['dbpath'])
# cursor = connection.cursor()
#
# if theme is 'reuters':
#     # Get titles
#     sql = 'SELECT title FROM article LIMIT 10'
#
# else:
#     # Get titles
#     sql = 'SELECT title FROM question LIMIT 10'
#
# cursor.execute(sql)
# result = cursor.fetchall()
# for row in result:
#     bow = prepro.vocabulary.doc2bow(utils.simple_preprocess(row[0]))
#     print('Title: %s - Topics: %s' % (row[0], tm.model[bow]))
