__author__ = 'Cordt Voigt'

from Topicmodel import Topicmodel
from Tagcloud import Tagcloud
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

theme = 'travel'
setting = {
    'theme': theme,
    'dbpath': '../../data/' + theme + '.db',
    'malletpath': '/usr/share/mallet-2.0.7/bin/mallet',
    'nooftopics': 10,
    'noofwordsfortopic': 50,
    'noofiterations': 50,
    'noofprocesses': 20,
    # Filter all words that appear in less documents
    'filter_less_than_no_of_documents': 5,
    # Filter all documents that appear in more than the given fraction of documents
    'filter_more_than_fraction_of_documents': 0.5,
    'create_tag_cloud': False}

if theme is 'reuters':
    setting['folderprefix'] = '../../data/' + theme + '/'
else:
    setting['folderprefix'] = '../../data/' + theme + '.stackexchange.com/'


# Topic model
tm = Topicmodel(setting)
tm.createmodel()

# Create tag cloud
if setting['create_tag_cloud']:
    Tagcloud.createtagcloud(tm, setting)

# Create similarities db if neccessary
directory = "../../results/" + setting['theme'] + "/model/"
filename = "similarities.db"
dbpath = ''.join([directory, filename])

if not os.path.isfile(dbpath):
    print("Similarities database does not exist yet, determining similarities and writing to database...")
    tm.determine_question_answer_distances()

# Determine avergae distances of answers, that are related to the question
tm.get_true_answers_distances(no_of_questions=-1)

# Determine the distance to the correct order of answers to a question, given by the upvote score, compared to the order
# given by the topic model
tm.compute_answer_order_metric(no_of_questions=-1)

