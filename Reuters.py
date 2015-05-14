__author__ = 'Cordt'

import sgmllib as sgm
import sqlite3
import os.path


class GS:
    def __init__(self, reuters_dir):
            self.reuters_dir = reuters_dir
            self.corpus = []

    def getrawcorpus(self):
        """Iterate over Reuters documents, yielding one document at a time."""
        for fname in os.listdir(self.reuters_dir):
            # read each document as one big string
            document = open(os.path.join(self.reuters_dir, fname)).read()
            # parse document into a list of utf8 tokens
            self.corpus.append(document)
        return self.corpus


class Importer:

    def __init__(self, setting):
        self.setting = setting
        self.corpus = []
        self.extractor = _Contentextractor()

        self.hastopic = []
        self.topics = [[]]
        self.places = [[]]
        self.elementid = []
        self.oldelementid = []
        self.creationdate = []
        self.title = []
        self.body = []
        self.lewissplit = []
        self.cgisplit = []

    def importdata(self):
        self._createdb()
        self._importtopicsandplaces(self.topics, self.places)
        self._importarticles()
        self._getcorpus()

        return self.corpus

    def _getcorpus(self):
        print("Loading articles...")
        sql = 'SELECT body FROM article'
        self.cursor.execute(sql)
        articles = self.cursor.fetchall()
        for row in articles:
            # Append document to corpus
            self.corpus.append(row[0])

        self.connection.close()

    def _createdb(self):
        # Create tables if database does not exist yet
        if not os.path.isfile(self.setting['dbpath']):
            print("Database does not exist yet, creating...")

            dataset = ''
            for index in range(0, 22):
                if index < 10:
                    number = '0' + str(index)
                else:
                    number = str(index)
                filename = '../../data/reuters/reut2-0' + number + '.sgm'
                rfile = open(filename, 'r')
                lines = rfile.readlines()
                for line in lines:
                    dataset += line
                rfile.close()
            for document in dataset:
                self.extractor.feed(document)
            self.extractor.close()

            self.hastopic = self.extractor.hastopic
            self.topics = self.extractor.topics
            self.places = self.extractor.places
            self.elementid = self.extractor.elementid
            self.oldelementid = self.extractor.oldelementid
            self.creationdate = self.extractor.creationdate
            self.title = self.extractor.title
            self.body = self.extractor.body
            self.lewissplit = self.extractor.lewissplit
            self.cgisplit = self.extractor.cgisplit

            # Database connection - instance variables
            self.connection = sqlite3.connect(self.setting['dbpath'])
            self.cursor = self.connection.cursor()

            # Create question table
            sql = 'CREATE TABLE article (elementId int, oldElementId int, creationDate int, title text, body text, \
                   lewisSplit int, cgiSplit int, hasTopic int)'
            self.cursor.execute(sql)

            # Create tag table
            sql = 'CREATE TABLE topic (elementId int, topicName text)'
            self.cursor.execute(sql)

            # Create postTag table
            sql = 'CREATE TABLE articleTopic (articleId int, topicId int)'
            self.cursor.execute(sql)

            # Create tag table
            sql = 'CREATE TABLE place (elementId int, placeName text)'
            self.cursor.execute(sql)

            # Create postTag table
            sql = 'CREATE TABLE articlePlace (articleId int, placeId int)'
            self.cursor.execute(sql)

            self.connection.commit()

        else:
            # Database connection - instance variables
            self.connection = sqlite3.connect(self.setting['dbpath'])
            self.cursor = self.connection.cursor()

            print("Database already exists, connecting...")

    def _importarticles(self):

        # Check whether database is empty
        sql = 'SELECT * FROM article'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        if len(result) != 0:
            print('Article table not empty, adding nothing')

        else:
            print('Importing articles into database')
            for index in range(0, len(self.title)):
                self._inserttopicsforarticle(self.topics[index], self.elementid[index])
                self._insertplacesforarticle(self.places[index], self.elementid[index])

                lewissplit = 1 if self.lewissplit[index] == 'TRAIN' else 0
                cgisplit = 1 if self.cgisplit[index] == 'TRAINING-SET' else 0
                hastopic = 1 if self.hastopic[index] == 'YES' else 0

                values = [self.elementid[index], self.oldelementid[index], self.creationdate[index],
                          self.title[index].decode('latin-1'), self.body[index].decode('latin-1'),
                          lewissplit, cgisplit, hastopic]

                self.cursor.execute('INSERT INTO article VALUES (?, ?,strftime(\'%s\', ?), ?, ?, ?, \
                                     ?, ?)', values)

                self.connection.commit()

    # Must be called before importposts
    def _importtopicsandplaces(self, topics, places):

        # Check whether database is empty
        sql = 'SELECT * FROM topic'
        self.cursor.execute(sql)
        result1 = self.cursor.fetchall()
        sql = 'SELECT * FROM place'
        self.cursor.execute(sql)
        result2 = self.cursor.fetchall()

        if len(result1) != 0 or len(result2) != 0:
            print('Topic or place table not empty, adding nothing')

        else:
            print('Importing topics and places into database')
            elementid = -1
            for topicset in topics:
                for topic in topicset:
                    self.cursor.execute('SELECT elementId FROM topic WHERE topicName=?', (topic, ))
                    if self.cursor.fetchone() is None:
                        elementid += 1
                        values = [elementid, topic]
                        self.cursor.execute('INSERT INTO topic VALUES (?, ?)', values)
            self.connection.commit()

            elementid = -1
            for placeset in places:
                for place in placeset:
                    self.cursor.execute('SELECT elementId FROM place WHERE placeName=?', (place, ))
                    if self.cursor.fetchone() is None:
                        elementid += 1
                        values = [elementid, place]
                        self.cursor.execute('INSERT INTO place VALUES (?, ?)', values)
            self.connection.commit()

    def _inserttopicsforarticle(self, topics, articleid):
        for topic in topics:
            self.cursor.execute('SELECT elementId FROM topic WHERE topicName=?', (topic, ))
            result = self.cursor.fetchone()
            if len(result) != 0:
                values = [articleid, result[0]]
                self.cursor.execute('INSERT INTO articleTopic VALUES (?, ?)', values)
            else:
                print('No topic record found for %s' % topic)

    def _insertplacesforarticle(self, places, articleid):
        for place in places:
            self.cursor.execute('SELECT elementId FROM place WHERE placeName=?', (place, ))
            result = self.cursor.fetchone()
            if len(result) != 0:
                values = [articleid, result[0]]
                self.cursor.execute('INSERT INTO articlePlace VALUES (?, ?)', values)
            else:
                print('No place record found for %s' % place)


class _Contentextractor(sgm.SGMLParser):

    def __init__(self, verbose=0):
        sgm.SGMLParser.__init__(self, verbose)
        self.hastopic = []
        self.topics = None
        self.places = None
        self.elementid = []
        self.oldelementid = []
        self.creationdate = []
        self.title = []
        self.body = []
        self.lewissplit = []
        self.cgisplit = []

        self.data = None
        self.index = 0
        self.in_topics = False
        self.in_places = False

    def handle_data(self, data):
        if self.data is not None:
            self.data.append(data)

    def start_reuters(self, attrs):
        self.hastopic.append(attrs[0][1])
        self.lewissplit.append(attrs[1][1])
        self.cgisplit.append(attrs[2][1])
        self.oldelementid.append(attrs[3][1])
        self.elementid.append(attrs[4][1])

        self.creationdate.append('')
        self.title.append('')
        self.body.append('')

    def end_reuters(self):
        self.index += 1

    def unknown_starttag(self, tag, attrs):
        self.data = []
        if tag == 'topics':
            self.in_topics = True
            if self.topics is None:
                self.topics = [[]]
            else:
                self.topics.append([])

        if tag == 'places':
            self.in_places = True
            if self.places is None:
                self.places = [[]]
            else:
                self.places.append([])

    def unknown_endtag(self, tag):

        if tag == 'topics':
            self.in_topics = False

        if tag == 'places':
            self.in_places = False

        if tag == 'date':
            self.creationdate[self.index] = "".join(self.data)

        if tag == 'title':
            self.title[self.index] = "".join(self.data)

        if tag == 'body':
            self.body[self.index] = "".join(self.data)

        if tag == 'd':
            if self.in_topics:
                self.topics[self.index].append("".join(self.data))
            elif self.in_places:
                self.places[self.index].append("".join(self.data))

        self.data = None