__author__ = 'Cordt Voigt'

from Topicmodel import Topicmodel
from Tagcloud import Tagcloud
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

theme = 'beer'
setting = {
    'theme': theme,
    'dbpath': '../../data/' + theme + '.db',
    'malletpath': '/usr/share/mallet-2.0.7/bin/mallet',
    'nooftopics': 10,
    'noofwordsfortopic': 50,
    'noofiterations': 500,
    'noofprocesses': 20,
    'histogramcut': 20.0}

if theme is 'reuters':
    setting['folderprefix'] = '../../data/' + theme + '/'
else:
    setting['folderprefix'] = '../../data/' + theme + '.stackexchange.com/'


# Topic model
tm = Topicmodel(setting)
tm.createmodel()

Tagcloud.createtagcloud(tm, setting)


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

