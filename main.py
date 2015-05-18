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

tm.determine_document_similarities()
# Tagcloud.createtagcloud(tm, setting)

