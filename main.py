__author__ = 'Cordt Voigt'

from Topicmodel import Topicmodel
from Tagcloud import Tagcloud
import logging
import os
import sys

if len(sys.argv) > 1:
    theme = sys.argv[1]
else:
    print('No theme provided')
    sys.exit(0)

logging.basicConfig(filename='/srv/cordt-mt/log/' + theme + '.log',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

setting = {
    'theme': theme,

    # 'dbpath': '../../data/' + theme + '.db',
    'dbpath': '/srv/cordt-mt/data/' + theme + '.db',

    # 'resultfolder': '/Users/Cordt/Documents/results/',
    'resultfolder': '/srv/cordt-mt/results/',

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
    'distance_metric': 2}

if theme is 'reuters':
    # setting['folderprefix'] = '../../data/' + theme + '/'
    setting['folderprefix'] = '/srv/cordt-mt/data/' + theme + '/'
else:
    # setting['folderprefix'] = '../../data/' + theme + '.stackexchange.com/'
    setting['folderprefix'] = '/srv/cordt-mt/data/' + theme + '.stackexchange.com/'

for mIndex in range(2, 4, 1):
    setting['distance_metric'] = mIndex
    for index in range(1, 4, 1):
        setting['data_for_model'] = index

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
            logging.info("Document lengths database does not exist yet, determining lengths and writing to database...")
            tm.determine_answer_lengths()

        # Determine avergae distances of answers, that are related to the question
        tm.get_true_answers_distances(no_of_questions=-1)

        # Determine the distance to the correct order of answers to a question, given by the upvote score,
        # compared to the order
        # given by the topic model
        tm.compute_answer_order_metric(no_of_questions=-1)

        tm.compute_answer_length_impact(no_of_questions=-1)
