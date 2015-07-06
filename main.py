__author__ = 'Cordt Voigt'

from Topicmodel import Topicmodel
from Tagcloud import Tagcloud
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

theme = 'islam'
setting = {
    'theme': theme,
    'dbpath': '../../data/' + theme + '.db',
    'resultfolder': '/Users/Cordt/Documents/results/',

    'malletpath': '/usr/share/mallet-2.0.7/bin/mallet',

    'nooftopics': 100,
    'noofwordsfortopic': 100,
    'noofiterations': 1000,
    'noofprocesses': 20,

    # Filter all words that appear in less documents
    'filter_less_than_no_of_documents': 5,

    # Filter all documents that appear in more than the given fraction of documents
    'filter_more_than_fraction_of_documents': 0.5,

    'create_tag_cloud': False,
    'clean_similarities_table': False,

    # Data used for topic model
    # 1: Questions, 2: Answers, 3: Both
    'data_for_model': 1,

    # Used distance metric for the permutations
    # 1: Exact match distance
    # 2: Deviation distance
    # 3: Squared deviation distance
    'distance_metric': 1}

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

# Create similarities tables
tm.determine_question_answer_distances()

# Create document lengths db if neccessary
directory = setting['resultfolder'] + setting['theme'] + "/model/"
filename = "lengths.db"
dbpath = ''.join([directory, filename])

if not os.path.isfile(dbpath):
    print("Document lengths database does not exist yet, determining lengths and writing to database...")
    tm.determine_answer_lengths()

# Determine avergae distances of answers, that are related to the question
tm.get_true_answers_distances(no_of_questions=-1)

# Determine the distance to the correct order of answers to a question, given by the upvote score, compared to the order
# given by the topic model
tm.compute_answer_order_metric(no_of_questions=-1)

tm.compute_answer_length_impact(no_of_questions=-1)
